#include "darknet.h"

void train_tracker( char *cfgfile, char *weightfile)
{
  char *folders = "/local_home/autonomy_hands_temporal/list/list.txt";
  char *backup_directory = "/local_home/backup/";
  srand(time(0));
  char *base = basecfg(cfgfile);
  printf("%s\n", base);
  float avg_loss = -1;
  network net = parse_network_cfg(cfgfile);
  if(weightfile){
    load_weights(&net, weightfile);
  }
  printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
  int imgs = net.batch*net.subdivisions*net.time_steps;
  int batch = net.batch;
  int steps = net.time_steps;
  int i = *net.seen/net.batch;
  int streams = batch/steps;
  data train, buffer;

  layer l = net.layers[net.n - 1];

  int side = l.side;
  int classes = l.classes;
  float jitter = l.jitter;

  list *flist = get_paths(folders);
  char **folder_paths = (char **)list_to_array(flist);
  load_args args = {0};
  args.w = net.w;
  args.h = net.h;
  args.paths = folder_paths;
  args.n = batch;
  args.m = flist->size;
  args.steps = steps;
  args.classes = classes;
  args.jitter = jitter;
  args.num_boxes = side;
  args.d = &buffer;
  args.type = TRACKER_DATA;

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
    strListMap *seq = malloc(sizeof(strListMap));
    seq->folder = folder;
    seq->frames = frame_paths;
    seq->count = frames->size;
    tsearch(seq, args.sequences, mapFind);
  }

  pthread_t load_thread = load_data_in_thread(args);
  clock_t time;
  while(get_current_batch(net) < net.max_batches)
  {
    i += 1;
    time=clock();
    pthread_join(load_thread, 0);
    train = buffer;
    load_thread = load_data_in_thread(args);

    printf("Loaded: %lf seconds\n", sec(clock()-time));

    time=clock();
    float loss = train_network(net, train);
    if (avg_loss < 0) avg_loss = loss;
    avg_loss = avg_loss*.9 + loss*.1;

    printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
    if(i%1000==0 || (i < 1000 && i%100 == 0)){
      char buff[256];
      sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
      save_weights(net, buff);
    }
    free_data(train);
  }
  char buff[256];
  sprintf(buff, "%s/%s_final.weights", backup_directory, base);
  save_weights(net, buff);
}

void run_tracker(int argc, char **argv)
{
  char *prefix = find_char_arg(argc, argv, "-prefix", 0);
  float thresh = find_float_arg(argc, argv, "-thresh", .2);
  int cam_index = find_int_arg(argc, argv, "-c", 0);
  int frame_skip = find_int_arg(argc, argv, "-s", 0);
  if(argc < 4){
    fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
    return;
  }

  int avg = find_int_arg(argc, argv, "-avg", 1);
  char *cfg = argv[3];
  char *weights = (argc > 4) ? argv[4] : 0;
  char *filename = (argc > 5) ? argv[5]: 0;
//  if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
  /*else */if(0==strcmp(argv[2], "train")) train_tracker(cfg, weights);
//  else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
//  else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
//  else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, 20, frame_skip, prefix, avg, .5, 0,0,0,0);
}
