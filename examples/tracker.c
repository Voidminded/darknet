#include "darknet.h"

void train_tracker(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
  list *options = read_data_cfg(datacfg);
  char *folders = option_find_str(options, "train", "data/train.list");
  char *backup_directory = option_find_str(options, "backup", "/backup/");

  srand(time(0));
  char *base = basecfg(cfgfile);
  printf("%s\n", base);
  float avg_loss = -1;
  network *nets = calloc(ngpus, sizeof(network));

  srand(time(0));
  int seed = rand();
  int i;
  for(i = 0; i < ngpus; ++i){
    srand(seed);
#ifdef GPU
    cuda_set_device(gpus[i]);
#endif
    nets[i] = parse_network_cfg(cfgfile);
    if(weightfile){
      load_weights(&nets[i], weightfile);
    }
    if(clear) *nets[i].seen = 0;
    nets[i].learning_rate *= ngpus;
  }
  srand(time(0));
  network net = nets[0];

  printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
  int imgs = net.batch *net.subdivisions *ngpus;
  int batch = net.batch /net.time_steps;
//  printf("batch : %d\n", net.batch);
  int steps = net.time_steps;
  //  int streams = batch/steps;
  data train, buffer;

  layer l = net.layers[net.n - 1];

  int classes = l.classes;
  float jitter = l.jitter;

  list *flist = get_paths(folders);
  char **folder_paths = (char **)list_to_array(flist);
//  void *root = 0;
  load_args args = {0};
  args.w = net.w;
  args.h = net.h;
  args.paths = folder_paths;
  args.n = batch;
  args.m = flist->size;
  args.steps = steps;
  args.classes = classes;
  args.jitter = jitter;
  args.num_boxes = l.max_boxes;
  args.d = &buffer;
  args.type = TRACKER_DATA;
  args.threads = 8;
//  args.sequences = root;
  args.sequences = calloc( flist->size, sizeof(char**));
  args.seq_frames = calloc( flist->size, sizeof(int));
  args.angle = net.angle;
  args.exposure = net.exposure;
  args.saturation = net.saturation;
  args.hue = net.hue;

  int j;
  for( j=0; j < flist->size; ++j)
  {
    char *folder = folder_paths[j];
    list *frames = get_paths(folder);
    char **frame_paths = (char**)list_to_array(frames);
//    strListMap *seq = malloc(sizeof(strListMap));
//    seq->folder = folder;
//    seq->frames = frame_paths;
//    seq->count = frames->size;
//    tsearch(seq, &args.sequences, compar);
    args.sequences[j] = frame_paths;
    args.seq_frames[j] = frames->size;
  }
  pthread_t load_thread = load_data(args);
  clock_t time;
  int count = 0;
  while(get_current_batch(net) < net.max_batches)
  {
    if(l.random && count++%10 == 0){
      printf("Resizing\n");
      int dim = (rand() % 10 + 10) * 32;
      if (get_current_batch(net)+200 > net.max_batches) dim = 608;
      //int dim = (rand() % 4 + 16) * 32;
      printf("%d\n", dim);
      args.w = dim;
      args.h = dim;

      pthread_join(load_thread, 0);
      train = buffer;
      free_data(train);
      load_thread = load_data(args);

      for(i = 0; i < ngpus; ++i){
        resize_network(nets + i, dim, dim);
      }
      net = nets[0];
    }
    time=clock();
    pthread_join(load_thread, 0);
    train = buffer;
    load_thread = load_data(args);

    printf("Loaded: %lf seconds\n", sec(clock()-time));

    time=clock();
    float loss = 0;
#ifdef GPU
    if(ngpus == 1){
      loss = train_network(net, train);
    } else {
      loss = train_networks(nets, ngpus, train, 4);
    }
#else
    loss = train_network(net, train);
#endif
    if (avg_loss < 0) avg_loss = loss;
    avg_loss = avg_loss*.9 + loss*.1;

    i = get_current_batch(net);
    printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
    if(i%1000==0){
#ifdef GPU
      if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
      char buff[256];
      sprintf(buff, "%s/%s.backup", backup_directory, base);
      save_weights(net, buff);
    }
    if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
      if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
      char buff[256];
      sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
      save_weights(net, buff);
    }
    free_data(train);
  }
#ifdef GPU
  if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
  char buff[256];
  sprintf(buff, "%s/%s_final.weights", backup_directory, base);
  save_weights(net, buff);
//  free(root);
}

void run_tracker(int argc, char **argv)
{
  char *prefix = find_char_arg(argc, argv, "-prefix", 0);
  float thresh = find_float_arg(argc, argv, "-thresh", .24);
  float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
  int cam_index = find_int_arg(argc, argv, "-c", 0);
  int frame_skip = find_int_arg(argc, argv, "-s", 0);
  int avg = find_int_arg(argc, argv, "-avg", 3);
  if(argc < 4){
    fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
    return;
  }
  char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
  char *outfile = find_char_arg(argc, argv, "-out", 0);
  int *gpus = 0;
  int gpu = 0;
  int ngpus = 0;
  if(gpu_list){
    printf("%s\n", gpu_list);
    int len = strlen(gpu_list);
    ngpus = 1;
    int i;
    for(i = 0; i < len; ++i){
      if (gpu_list[i] == ',') ++ngpus;
    }
    gpus = calloc(ngpus, sizeof(int));
    for(i = 0; i < ngpus; ++i){
      gpus[i] = atoi(gpu_list);
      gpu_list = strchr(gpu_list, ',')+1;
    }
  } else {
    gpu = gpu_index;
    gpus = &gpu;
    ngpus = 1;
  }

  int clear = find_arg(argc, argv, "-clear");
  int fullscreen = find_arg(argc, argv, "-fullscreen");
  int width = find_int_arg(argc, argv, "-w", 0);
  int height = find_int_arg(argc, argv, "-h", 0);
  int fps = find_int_arg(argc, argv, "-fps", 0);

  char *datacfg = argv[3];
  char *cfg = argv[4];
  char *weights = (argc > 5) ? argv[5] : 0;
  char *filename = (argc > 6) ? argv[6]: 0;
  //  if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
  /*else */if(0==strcmp(argv[2], "train")) train_tracker(datacfg, cfg, weights, gpus, ngpus, clear);
  //  else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
  //  else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
  //  else if(0==strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights);
  //  else if(0==strcmp(argv[2], "demo")) {
  //      list *options = read_data_cfg(datacfg);
  //      int classes = option_find_int(options, "classes", 20);
  //      char *name_list = option_find_str(options, "names", "data/names.list");
  //      char **names = get_labels(name_list);
  //      demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
  //  }
}
