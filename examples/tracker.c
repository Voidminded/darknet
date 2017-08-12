#include "darknet.h"

void train_tracker(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
  list *options = read_data_cfg(datacfg);
  char *folders = option_find_str(options, "train", "data/train.list");
  char *backup_directory = option_find_str(options, "backup", "/backup/");
  char *train_csv_name = "train.csv";

  FILE *train_csv_file = fopen(train_csv_name,"w+");
  fprintf(train_csv_file,"Batch,Loss,Avg,Rate,Images\n");
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
  args.n = batch*net.subdivisions*ngpus;
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
    fprintf( train_csv_file, "%d,%f,%f,%f,%d\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), i*imgs);
    fflush( train_csv_file);
    if(i%1000==0){
#ifdef GPU
      if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
      char buff[256];
      sprintf(buff, "%s/%s.backup", backup_directory, base);
      save_weights(net, buff);
    }
    if(i%10000==0 || (i < 10000 && i%1000 == 0) || (i < 1000 && i%100 == 0)){
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
  fclose(train_csv_file);
//  free(root);
}

typedef struct {
    float *x;
    float *y;
} float_pair;

float_pair convert_mat_to_pair(data d, network net, int batch, int steps)
{
  int b;
//  assert(net.batch == steps + 1);
  float_pair p = {0};
  p.x = calloc(net.batch*batch*d.X.cols, sizeof(float));
  p.y = calloc(net.batch*batch*d.y.cols, sizeof(float));
  for(b = 0; b < batch; ++b){
    int i;
    for(i = 0; i < steps; ++i){
      memcpy(p.x + i*d.X.cols + b*steps*d.X.cols, d.X.vals[i+ b*steps], d.X.cols*sizeof(float));
      memcpy(p.y + i*d.y.cols + b*steps*d.y.cols, d.y.vals[i+ b*steps], d.y.cols*sizeof(float));
    }
  }
  return p;
}

void train_vid_tracker(char *datacfg, char *cfgfile, char *weightfile, int clear)
{
  list *options = read_data_cfg(datacfg);
  char *folders = option_find_str(options, "train", "data/train.list");
  char *backup_directory = option_find_str(options, "backup", "/backup/");
  char *train_csv_name = "train.csv";

  FILE *train_csv_file = fopen(train_csv_name,"w+");
  fprintf(train_csv_file,"Batch,Loss,Avg,Rate,Images\n");
  srand(time(0));
  char *base = basecfg(cfgfile);
  printf("%s\n", base);
  float avg_loss = -1;
  network net = parse_network_cfg(cfgfile);
  if(weightfile){
      load_weights(&net, weightfile);
  }
  printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
  int imgs = net.batch *net.subdivisions;
  int i = *net.seen/imgs;

  list *flist = get_paths(folders);
  char **folder_paths = (char **)list_to_array(flist);
  clock_t time;
  int batch = net.batch /net.time_steps;
  int steps = net.time_steps;
  layer l = net.layers[net.n - 1];

  int classes = l.classes;
  float jitter = l.jitter;

  data train, buffer;
  load_args args = {0};
  args.w = net.w;
  args.h = net.h;
  args.paths = folder_paths;
  args.n = batch*net.subdivisions;
  args.m = flist->size;
  args.steps = steps;
  args.classes = classes;
  args.jitter = jitter;
  args.num_boxes = l.max_boxes;
  args.d = &buffer;
  args.type = TRACKER_DATA;
  args.threads = 8;
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
    args.sequences[j] = frame_paths;
    args.seq_frames[j] = frames->size;
  }
  pthread_t load_thread = load_data(args);
  while(get_current_batch(net) < net.max_batches)
  {
    time=clock();

    pthread_join(load_thread, 0);
    train = buffer;
    load_thread = load_data(args);
    // float_pair p = convert_mat_to_pair(train, net, batch, steps);

    // copy_cpu(net.inputs*net.batch, p.x, 1, net.input, 1);
    // copy_cpu(net.truths*net.batch, p.y, 1, net.truth, 1);
    printf("Loaded: %lf seconds\n", sec(clock()-time));

    time=clock();
    float loss = train_network(net, train);
    if (avg_loss < 0) avg_loss = loss;
    avg_loss = avg_loss*.9 + loss*.1;
    // free(p.x);
    // free(p.y);
    i = get_current_batch(net);
    printf("%d: %f, %f avg, %g rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
    fprintf( train_csv_file, "%d,%f,%f,%f,%d\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), i*imgs);
    fflush( train_csv_file);
    if(i%1000==0){
      char buff[256];
      sprintf(buff, "%s/%s.backup", backup_directory, base);
      save_weights(net, buff);
    }
    if(i%10000==0 || (i < 10000 && i%1000 == 0) || (i < 1000 && i%100 == 0)){
      char buff[256];
      sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
      save_weights(net, buff);
    }
    free_data(train);
  }
  char buff[256];
  sprintf(buff, "%s/%s_final.weights", backup_directory, base);
  save_weights(net, buff);
  fclose(train_csv_file);
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
  else if(0==strcmp(argv[2], "train2")) train_vid_tracker(datacfg, cfg, weights, clear);
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
