#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
#include "image.h"

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

void train_segmenter(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int display)
{
    int i;

    char *train_csv_name = "train.csv";

    FILE *train_csv_file = fopen(train_csv_name,"w+");
    fprintf(train_csv_file,"Batche,Epoch,Loss,Avg,Rate\n");

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];
    image pred = get_network_image(net);

    int div = net->w/pred.w;
    //printf("Div = %d, %d, %d\n", div, net->w, pred.w); 
    assert(pred.w * div == net->w);
    assert(pred.h * div == net->h);

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_list = option_find_str(options, "train", "data/train.list");

    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 4;
    args.scale = div;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;
    args.classes = 1;

    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.type = SEGMENTATION_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        double time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

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
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        fprintf( train_csv_file, "%d,%f,%f,%f,%g\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net));
        fflush( train_csv_file);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if( get_current_batch(net)%10 == 0){
            image trth = float_to_image(net->w/div, net->h/div, 12, train.y.vals[net->batch*(net->subdivisions-1)]);
           // image tr = collapse_birds_layers( trth, 1);
           // save_image_16( tr, "truth");
           // image im = collapse_image_layers( float_to_image(net->w, net->h, 9, train.X.vals[net->batch*(net->subdivisions-1)]), 1);
           // save_image(im, "input");
           // free_image( im);
           // image pr = collapse_birds_layers(pred, 1);
           // save_image_16( pr, "pred");
           // image dist = image_distance( tr, pr);
           // save_image_16(dist, "dist");
           // image diff = image_diff( tr, pr);
           // save_image_16(diff, "diff");
            //------------------------
            //for debug:
            image spc = make_image( trth.w*3+6, trth.h*12+33, 3);
            fill_image( spc, 1.);
            image tmp = make_empty_image(0,0,0);
            int ind_spc;
            for( ind_spc = 0; ind_spc <3; ind_spc++)
            {
                tmp = get_image_layer( trth, ind_spc);
                place_image( tmp, trth.w, trth.h,  0, ind_spc*(trth.h+3), spc);
                free_image( tmp);
            }
            image msk = get_image_layer( pred, 0);
            tmp = bird_to_rgb( msk, 0);
            place_image( tmp, pred.w, pred.h,  pred.w+3, 0, spc);
            place_image( tmp, pred.w, pred.h,  2*(pred.w+3), 0, spc);
            free_image( tmp);
            for( ind_spc = 1; ind_spc <12; ind_spc++)
            {
                image t = get_image_layer( pred, ind_spc);
                tmp = bird_to_rgb( t, ind_spc);
                place_image( tmp, pred.w, pred.h, 2*(pred.w+3), ind_spc*(pred.h+3), spc);
                free_image( tmp);
                mul_cpu( pred.w*pred.h, msk.data, 1, t.data, 1);
                tmp = bird_to_rgb( t, ind_spc);
                place_image( tmp, pred.w, pred.h,  pred.w+3, ind_spc*(pred.h+3), spc);
                free_image( t);
                free_image( tmp);
            }
            char f[128];
            sprintf( f, "./spc/%d_soft", get_current_batch(net)/10);
            save_image(spc, f);

            //-------------------------------


           // free_image( tr);
           // free_image( pr);
           // free_image( diff);
           // free_image( dist);
            free_image( spc);
        }
        if( get_current_batch(net)%100 == 0){
            char buff[256];
           // char file_name[21];
           // image tr = collapse_birds_layers( float_to_image(net->w/div, net->h/div, 12, train.y.vals[net->batch*(net->subdivisions-1)]), 1);
           // sprintf( file_name, "%d_truth", get_current_batch(net)/100);
           // save_image_16(tr, file_name);
           // image im = collapse_image_layers( float_to_image(net->w, net->h, 9, train.X.vals[net->batch*(net->subdivisions-1)]), 1);
           // sprintf( file_name, "%d_input", get_current_batch(net)/100);
           // save_image(im, file_name);
           // free_image( im);
           // sprintf( file_name, "%d", get_current_batch(net)/100);
           // image pr = collapse_birds_layers(pred, 1);
           // save_image_16(pr, file_name);
           // image dist = image_distance( tr, pr);
           // sprintf( file_name, "%d_dist", get_current_batch(net)/100);
           // save_image_16(dist, file_name);
           // image diff = image_diff( tr, pr);
           // sprintf( file_name, "%d_diff", get_current_batch(net)/100);
           // save_image_16(diff, file_name);
           // free_image( tr);
           // free_image( pr);
           // free_image( diff);
           // free_image( dist);
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
    fclose(train_csv_file);
}

void dual_train_segmenter(char *datacfg, char *cfgfile1, char *cfgfile2, char *weightfile1, char *weightfile2,int *gpus, int ngpus, int clear)
{
    int i;


    float avg_loss[2] = {-1, -1};
    char *cfgfile[2] = { cfgfile1, cfgfile2};
    char *weightfile[2] = { weightfile1, weightfile2};
    char *base[2] = { basecfg(cfgfile[0]),  basecfg(cfgfile[1])};
    char train_csv_name[256];
    FILE **train_csv_file = calloc( 2, sizeof( FILE*));
    for( i = 0; i < 2; ++i)
    {
        sprintf( train_csv_name, "%s_train.csv", base[i]);
        train_csv_file[i] = fopen(train_csv_name,"a");
        fprintf(train_csv_file[i],"Batche,Epoch,Loss,Avg,Rate\n");
    }
    printf("%s & %s\n", base[0], base[1]);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile[i], weightfile[i], clear);
     //   nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];
    image pred[2] = { get_network_image(nets[0]), get_network_image(nets[1])};

    int div = nets[0]->w/pred[0].w;
    //printf("Div = %d, %d, %d\n", div, net->w, pred.w); 
    assert(pred[0].w * div == nets[0]->w);
    assert(pred[0].h * div == nets[0]->h);

    div = nets[1]->w/pred[1].w;
    assert(pred[1].w * div == nets[1]->w);
    assert(pred[1].h * div == nets[1]->h);

    int imgs = net->batch * net->subdivisions;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_list = option_find_str(options, "train", "data/train.list");

    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 8;
    args.scale = div;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;
    args.classes = 1;

    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.type = SEGMENTATION_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        double time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float loss[2] = {0, 0};
        train_network_dual(nets, ngpus, train, loss);
        for( i = 0; i < ngpus; ++i){
            if(avg_loss[i] == -1) avg_loss[i] = loss[i];
            avg_loss[i] = avg_loss[i]*.9 + loss[i]*.1;
            printf("%s %ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", base[i], get_current_batch(net), (float)(*net->seen)/N, loss[i], avg_loss[i], get_current_rate(net), what_time_is_it_now()-time, *net->seen);
            fprintf( train_csv_file[i], "%d,%f,%f,%f,%g\n", get_current_batch(net), (float)(*net->seen)/N, loss[i], avg_loss[i], get_current_rate(net));
            fflush( train_csv_file[i]);
            if(*net->seen/N > epoch){
                epoch = *net->seen/N;
                char buff[256];
                sprintf(buff, "%s/%s_%d.weights",backup_directory,base[i], epoch);
                save_weights(net, buff);
            }
        }
        if( get_current_batch(net)%10 == 0){
            image trth = float_to_image(net->w/div, net->h/div, 3, train.y.vals[net->batch*(net->subdivisions-1)]);
            image spc = make_image( trth.w*5+2, trth.h*3+6, 1);
            fill_image( spc, 1.);
            image tmp = make_empty_image(0,0,0);
            int ind_spc;
            image msk = get_image_layer( pred[0], 0);
            place_image( msk, pred[0].w, pred[0].h,  0, 0, spc);
            place_image( msk, pred[0].w, pred[0].h,  pred[0].w+3, 0, spc);
            for( ind_spc = 1; ind_spc <3; ind_spc++)
            {
                image t = get_image_layer( pred[0], ind_spc);
                place_image( t, pred[0].w, pred[0].h,  0, ind_spc*(pred[0].h+3), spc);
                mul_cpu( pred[0].w*pred[0].h, msk.data, 1, t.data, 1);
                place_image( t, pred[0].w, pred[0].h,  pred[0].w+3, ind_spc*(pred[0].h+3), spc);
                free_image( t);
            }
            for( ind_spc = 0; ind_spc <3; ind_spc++)
            {
                image t = get_image_layer( trth, ind_spc);
                place_image( t, trth.w, trth.h,  2*(trth.w+3), ind_spc*(trth.h+3), spc);
                free_image( t);
            }
            free_image( msk);
            msk = get_image_layer( pred[1], 0);
            place_image( msk, pred[1].w, pred[1].h,  3*(pred[1].w+3), 0, spc);
            place_image( msk, pred[1].w, pred[1].h,  4*(pred[1].w+3), 0, spc);
            for( ind_spc = 1; ind_spc <3; ind_spc++)
            {
                image t = get_image_layer( pred[1], ind_spc);
                place_image( t, pred[1].w, pred[1].h,  4*(pred[1].w+3), ind_spc*(pred[0].h+3), spc);
                mul_cpu( pred[1].w*pred[1].h, msk.data, 1, t.data, 1);
                place_image( t, pred[1].w, pred[1].h,  3*(pred[1].w+3), ind_spc*(pred[0].h+3), spc);
                free_image( t);
            }
            char f[128];
            sprintf( f, "./dual/%d", get_current_batch(net)/10);
            save_image(spc, f);

            //-------------------------------


           // free_image( tr);
           // free_image( pr);
           // free_image( diff);
           // free_image( dist);
            free_image( spc);
        }
        if( get_current_batch(net)%100 == 0){
            char buff[256];
           // char file_name[21];
           // image tr = collapse_birds_layers( float_to_image(net->w/div, net->h/div, 12, train.y.vals[net->batch*(net->subdivisions-1)]), 1);
           // sprintf( file_name, "%d_truth", get_current_batch(net)/100);
           // save_image_16(tr, file_name);
           // image im = collapse_image_layers( float_to_image(net->w, net->h, 9, train.X.vals[net->batch*(net->subdivisions-1)]), 1);
           // sprintf( file_name, "%d_input", get_current_batch(net)/100);
           // save_image(im, file_name);
           // free_image( im);
           // sprintf( file_name, "%d", get_current_batch(net)/100);
           // image pr = collapse_birds_layers(pred, 1);
           // save_image_16(pr, file_name);
           // image dist = image_distance( tr, pr);
           // sprintf( file_name, "%d_dist", get_current_batch(net)/100);
           // save_image_16(dist, file_name);
           // image diff = image_diff( tr, pr);
           // sprintf( file_name, "%d_diff", get_current_batch(net)/100);
           // save_image_16(diff, file_name);
           // free_image( tr);
           // free_image( pr);
           // free_image( diff);
           // free_image( dist);
            sprintf(buff, "%s/%s.backup",backup_directory,base[0]);
            save_weights(nets[0], buff);
            sprintf(buff, "%s/%s.backup",backup_directory,base[1]);
            save_weights(nets[1], buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base[0]);
    save_weights(nets[0], buff);
    sprintf(buff, "%s/%s.weights", backup_directory, base[1]);
    save_weights(nets[1], buff);

    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    for( i = 0; i < ngpus; ++i){
        free_network(nets[i]);
        free(base[i]);
        fclose(train_csv_file[i]);
    }
}

void seq_predict_segmenter(char *datafile, char *cfg, char *weights, char *filename, int* gpus)
{
  cuda_set_device(gpus[0]);
  network *net = load_network(cfg, weights, 0);
  set_batch_network(net, 1);

  char buff[256];
  char *input = buff;
  list *plist = get_paths(filename);
  char **paths = (char **)list_to_array(plist);
  int n,m, w = net->w, h = net->h;
  image in = make_image( net->w, net->h, net->c);
  //image result = make_image( w*2+3, h*6+15, 3);
  image result = make_image( w*3+6, h*3+6, 3);
  fill_image( result, 1.);
  image im = make_empty_image(0,0,0);
  image sized = make_empty_image( 0,0,0);
  image gt = make_image( w,h,3);
  for( n=8; n < plist->size; ++n)
  {
    int dx = 0, dy = 0;
    for( m = 0; m < 9; ++m)
    {
      strncpy(input, paths[n-m], 256);
      im = load_image_color(input, 0, 0);
      sized = crop_image(im, dx, dy, net->w, net->h);
      memcpy( in.data + m*in.h*in.w, sized.data, sized.h*sized.w*sizeof( float));
      free_image( sized);
      free_image(im);
    }
    strncpy(input, paths[n], 256);
    find_replace(input, "image", "label/sp", input);
    find_replace(input, ".png", "_sp.png", input);
    im = load_image_color(input, 0, 0);
    sized = crop_image(im, dx, dy, net->w, net->h);
    memset( gt.data, 0, w*h*3*sizeof(float));
    int i_gt, j_gt;
    for( i_gt = 0; i_gt < w; ++i_gt)
      for( j_gt = 0; j_gt < h; ++j_gt)
        if( get_pixel( sized, i_gt, j_gt, 0)){
            set_pixel( gt, i_gt, j_gt, 0, 1.);
            get_pixel( sized, i_gt, j_gt, 0) > 0.9 ? set_pixel( gt, i_gt, j_gt, 1, 1.) : set_pixel( gt, i_gt, j_gt, 2, 1.);
            }
    float *X = in.data;
    network_predict(net, X);
    image pred = get_network_image(net);
    image tmp = make_empty_image(0,0,0);
    int ind_spc;
    image msk = get_image_layer( gt, 0);
    tmp = bird_to_rgb( msk, 0);
    place_image( tmp, pred.w, pred.h,  dx, dy, result);
    free_image( msk);
    msk = get_image_layer( pred, 0);
    tmp = bird_to_rgb( msk, 0);
    //place_image( tmp, pred.w, pred.h,  dx, dy, result);
    place_image( tmp, pred.w, pred.h,  dx+2*(w+3), dy, result);
    free_image( tmp);
    for( ind_spc = 1; ind_spc <3; ind_spc++)
    {
        //image t = get_image_layer( pred, ind_spc);
        image t = get_image_layer( gt, ind_spc);
        tmp = bird_to_rgb( t, ind_spc);
        place_image( tmp, pred.w, pred.h, dx, ind_spc*(h+3)+dy, result);
        free_image( tmp);
        free_image(t);
        t = get_image_layer( pred, ind_spc);
        //mul_cpu( pred.w*pred.h, msk.data, 1, t.data, 1);
        tmp = bird_to_rgb( t, ind_spc);
        place_image( tmp, pred.w, pred.h,  2*(w+3)+dx, ind_spc*(h+3)+dy, result);
        free_image( t);
        free_image( tmp);
    }
    int i_spc, j_spc;
    for( i_spc = 0; i_spc < pred.w; ++i_spc)
      for( j_spc = 0; j_spc < pred.h; ++j_spc)
      {
          float val_spc = get_pixel( pred,i_spc, j_spc, 0);
          int ind_spc = 0;
          if( get_pixel( pred,i_spc, j_spc, 1) > val_spc)
          {
              val_spc =  get_pixel( pred,i_spc, j_spc, 1);
              ind_spc = 1;
          }
          if( get_pixel( pred,i_spc, j_spc, 2) > val_spc)
          {
              val_spc =  get_pixel( pred,i_spc, j_spc, 2);
              ind_spc = 2;
          }
          set_pixel( result, pred.w+3+i_spc+dx, 0*(pred.h+3)+j_spc+dy, 0, 0.);
          set_pixel( result, pred.w+3+i_spc+dx, 1*(pred.h+3)+j_spc+dy, 0, 0.);
          set_pixel( result, pred.w+3+i_spc+dx, 2*(pred.h+3)+j_spc+dy, 0, 0.);
          set_pixel( result, pred.w+3+i_spc+dx, 0*(pred.h+3)+j_spc+dy, 1, 0.);
          set_pixel( result, pred.w+3+i_spc+dx, 1*(pred.h+3)+j_spc+dy, 1, 0.);
          set_pixel( result, pred.w+3+i_spc+dx, 2*(pred.h+3)+j_spc+dy, 1, 0.);
          set_pixel( result, pred.w+3+i_spc+dx, 0*(pred.h+3)+j_spc+dy, 2, 0.);
          set_pixel( result, pred.w+3+i_spc+dx, 1*(pred.h+3)+j_spc+dy, 2, 0.);
          set_pixel( result, pred.w+3+i_spc+dx, 2*(pred.h+3)+j_spc+dy, 2, 0.);
          set_pixel( result, pred.w+3+i_spc+dx, ind_spc*(pred.h+3)+j_spc+dy, 0, 1);
          set_pixel( result, pred.w+3+i_spc+dx, ind_spc*(pred.h+3)+j_spc+dy, 1, 1);
          set_pixel( result, pred.w+3+i_spc+dx, ind_spc*(pred.h+3)+j_spc+dy, 2, 1);

          //for( ind_spc = 0; ind_spc <3; ind_spc++)
          //{
          //    tmp = get_image_layer( pred, ind_spc);
          //    if( 
          //    place_image( tmp, pred.w, pred.h,  pred.w+3, ind_spc*(pred.h+3), spc);
          //    free_image( tmp);
          //}
      }
    char f[128];
    sprintf( f, "./result/%d", n);
    free_image( msk);
    //show_image(im, "orig", 1);
    //show_image(result, "pred", 0);
    char pred_name[256];
    strncpy(input, paths[n], 256);
    sprintf( pred_name, "results/pred_%s", basecfg( input));
    save_image(result, pred_name);
  }
  free_image(result);
}

void seq_crop_predict_segmenter(char *datafile, char *cfg, char *weights, char *filename)
{
  network *net = load_network(cfg, weights, 0);
  set_batch_network(net, 1);

  char buff[256];
  char *input = buff;
  list *plist = get_paths(filename);
  char **paths = (char **)list_to_array(plist);
  int n,m, w = 1920, h = 1080;
  image in = make_image( net->w, net->h, net->c);
  image result = make_image( w*2+3, h*6+15, 3);
  fill_image( result, 1.);
  image im = make_empty_image(0,0,0);
  image sized = make_empty_image( 0,0,0);
  for( n=8; n < plist->size; ++n)
  {
    int oi ,oj;
    for( oi = 0; oi < 4; ++oi)
    {
      for( oj = 0; oj < 2; oj++)
      {
        int dx, dy;
        switch( oi)
        {
          case 0:
            dx = 0;
            break;

          case 1:
            dx = 416;
            break;

          case 2:
            dx = 896;
            break;

          case 3:
            dx = 1312;
            break;
        }
        if( oj)
          dy = 472;
        else
          dy = 0;
        for( m = 0; m < 9; ++m)
        {
          strncpy(input, paths[n-m], 256);
          im = load_image_color(input, 0, 0);
          sized = crop_image(im, dx, dy, net->w, net->h);
          memcpy( in.data + m*in.h*in.w, sized.data, sized.h*sized.w*sizeof( float));
          free_image( sized);
          free_image(im);
        }
        float *X = in.data;
        network_predict(net, X);
        image pred = get_network_image(net);
        image tmp = make_empty_image(0,0,0);
        int ind_spc;
        image msk = get_image_layer( pred, 0);
        tmp = bird_to_rgb( msk, 0);
        place_image( tmp, pred.w, pred.h,  dx, dy, result);
        place_image( tmp, pred.w, pred.h,  dx+w+3, dy, result);
        free_image( tmp);
        for( ind_spc = 1; ind_spc <6; ind_spc++)
        {
            image t = get_image_layer( pred, ind_spc);
            tmp = bird_to_rgb( t, ind_spc);
            place_image( tmp, pred.w, pred.h, dx, ind_spc*(h+3)+dy, result);
            free_image( tmp);
            mul_cpu( pred.w*pred.h, msk.data, 1, t.data, 1);
            tmp = bird_to_rgb( t, ind_spc);
            place_image( tmp, pred.w, pred.h,  w+3+dx, ind_spc*(h+3)+dy, result);
            free_image( t);
            free_image( tmp);
        }
        char f[128];
        sprintf( f, "./result/%d", n);
        free_image( msk);
      }
    }
    //show_image(im, "orig", 1);
    //show_image(result, "pred", 0);
    char pred_name[256];
    strncpy(input, paths[n], 256);
    sprintf( pred_name, "results/pred_%s", basecfg( input));
    save_image(result, pred_name);
  }
  free_image(result);
}

void batch_predict_segmenter(char *datafile, char *cfg, char *weights, char *filename)
{
  network *net = load_network(cfg, weights, 0);
  set_batch_network(net, 1);

  char buff[256];
  char *input = buff;
  list *plist = get_paths(filename);
  char **paths = (char **)list_to_array(plist);
  int n;
  for( n=0; n < plist->size; ++n)
  {
    strncpy(input, paths[n], 256);
    image im = load_image_color(input, 0, 0);
    image result = make_image(im.w, im.h, im.c);
    int oi ,oj;
    for( oi = 0; oi < 4; ++oi)
    {
      for( oj = 0; oj < 2; oj++)
      {
        int dx, dy;
        switch( oi)
        {
          case 0:
            dx = 0;
            break;

          case 1:
            dx = 416;
            break;

          case 2:
            dx = 896;
            break;

          case 3:
            dx = 1312;
            break;
        }
        if( oj)
          dy = 472;
        else
          dy = 0;
        image sized = crop_image(im, dx, dy, net->w, net->h);
        float *X = sized.data;
        network_predict(net, X);
        image pred = get_network_image(net);
        int i, j, k;
        float val;
        for(k = 0; k < 3; ++k){
          for(j = 0; j < pred.h; ++j){
            for(i = 0; i < pred.w; ++i){
              int r = j + dy;
              int c = i + dx;
              val = get_pixel(pred, i, j, k);
              if( get_pixel(pred, i, j, 3) > 0.5)
              {
                if( k == 2)// B -> c[2] -> x
                  val = fmin( 1.0, fmax( 0, (val * ( (float) net->w / (float)im.w)) + ((float)( dx - 1) / (float)im.w)));
                else if( k == 1) // G -> c[1] -> y
                  val = fmin( 1.0, fmax( 0, (val * ( (float) net->h / (float)im.h)) + ((float)( dy - 1) / (float)im.h)));
                float cur_val = get_pixel(result, c, r, k);
                if( cur_val > 2*1e-5)
                  val = (val+cur_val)/2.;
              }
              set_pixel(result, c, r, k, val);
            }
          }
        }
        free_image(sized);
      }
    }
    //show_image(im, "orig", 1);
    //show_image(result, "pred", 0);
    char pred_name[256];
    sprintf( pred_name, "results/pred_%s", basecfg( input));
    save_image_16(result, pred_name);
    free_image(im);
    free_image(result);
  }

}

void predict_segmenter(char *datafile, char *cfg, char *weights, char *filename)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    srand(2222222);

    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image result = make_image(im.w, im.h, im.c);
        int oi ,oj;
        for( oi = 0; oi < 4; ++oi)
        {
          for( oj = 0; oj < 2; oj++)
          {
            int dx, dy;
            switch( oi)
            {
              case 0:
                dx = 0;
                break;

              case 1:
                dx = 416;
                break;

              case 2:
                dx = 896;
                break;

              case 3:
                dx = 1312;
                break;
            }
            if( oj)
              dy = 472;
            else
              dy = 0;
            image sized = crop_image(im, dx, dy, net->w, net->h);
            float *X = sized.data;
            network_predict(net, X);
            image pred = get_network_image(net);
            int i, j, k;
            float val;
            for(k = 0; k < 3; ++k){
              for(j = 0; j < pred.h; ++j){
                for(i = 0; i < pred.w; ++i){
                  int r = j + dy;
                  int c = i + dx;
                  val = get_pixel(pred, i, j, k);
                  if( get_pixel(pred, i, j, 3) > 0.5)
                  {
                    if( k == 2)// B -> c[2] -> x
                      val = fmin( 1.0, fmax( 0, (val * ( (float) net->w / (float)im.w)) + ((float)( dx - 1) / (float)im.w)));
                    else if( k == 1) // G -> c[1] -> y
                      val = fmin( 1.0, fmax( 0, (val * ( (float) net->h / (float)im.h)) + ((float)( dy - 1) / (float)im.h)));
                    float cur_val = get_pixel(result, c, r, k);
                    if( cur_val > 2*1e-5)
                      val = (val+cur_val)/2.;
                  }
                  set_pixel(result, c, r, k, val);
                }
              }
            }
          }
        }
        //show_image(im, "orig", 1);
        //show_image(result, "pred", 0);
        char pred_name[256];
        sprintf( pred_name, "results/pred_%s", basecfg( filename));
        save_image_16(result, pred_name);
        free_image(im);
        free_image(result);
        if (filename) break;
    }
}

void predict_visualize_single_crop(char *datafile, char *cfg, char *weights, char *filename)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);

    char buff[256];
    char *input = buff;
    if(filename){
        strncpy(input, filename, 256);
    }else{
        printf("Enter Image Path: ");
        fflush(stdout);
        input = fgets(input, 256, stdin);
        if(!input) return;
        strtok(input, "\n");
    }
    image im = load_image_color(input, 0, 0);
    float *X = im.data;
    network_predict(net, X);
    int i;
    for(i = net->n-4; i >= 0; --i){
      image m = get_network_image_layer(net, i);
      char buff[256];
      sprintf(buff, "Layer %d Output", i);
      image collapsed = collapse_image_layers(m, 1);
      save_image( collapsed, buff);
    }
    visualize_network(net);
    image pred = get_network_image(net);
    show_image(im, "orig", 1);
    show_image(pred, "pred", 0);
    char pred_name[256];
    sprintf( pred_name, "results/pred_%s", basecfg( filename));
    save_image_16(pred, pred_name);
    free_image(im);
    free_image(pred);
}

void demo_segmenter(char *datacfg, char *cfg, char *weights, int cam_index, const char *filename)
{
#ifdef OPENCV
    printf("Classifier Demo\n");
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow("Segmenter", CV_WINDOW_NORMAL); 
    cvResizeWindow("Segmenter", 512, 512);
    float fps = 0;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        image in_s = letterbox_image(in, net->w, net->h);

        network_predict(net, in_s.data);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        image pred = get_network_image(net);
        image prmask = mask_to_rgb(pred);
        show_image(prmask, "Segmenter", 10);
        
        free_image(in_s);
        free_image(in);
        free_image(prmask);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}


void run_segmenter(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
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

    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int clear = find_arg(argc, argv, "-clear");
    int display = find_arg(argc, argv, "-display");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    char *weight2 = (argc > 7) ? argv[7]: 0;
    if(0==strcmp(argv[2], "test")) predict_segmenter(data, cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_segmenter(data, cfg, weights, gpus, ngpus, clear, display);
    else if(0==strcmp(argv[2], "dualtrain")) dual_train_segmenter(data, cfg, weights, filename, weight2, gpus, ngpus, clear); // weights: cfg2, filename: weight1
    else if(0==strcmp(argv[2], "demo")) demo_segmenter(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "batchtest")) seq_predict_segmenter(data, cfg, weights, filename, gpus);
    else if(0==strcmp(argv[2], "visualize")) predict_visualize_single_crop(data, cfg, weights, filename);
}


