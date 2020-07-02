#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
#include <blas.h>
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

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));
    char train_csv_name[256];
    sprintf( train_csv_name, "%s_train.csv", base);
    FILE *train_csv_file = fopen(train_csv_name,"a");
    // Reduced channels
    //fprintf(train_csv_file,"Birdness,Jackdaw,Rook,Dx,Dy,Dz,Qx,Qy,Qz,Qw,Sin(f),Cos(f),Batche,Epoch,Loss,Avg,Rate\n");
    fprintf(train_csv_file,"Birdness,Jackdaw,Rook,Dx,Dy,Dz,Batche,Epoch,Loss,Avg,Rate\n");

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
    net->log = train_csv_file;

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
        fprintf( train_csv_file, ",%d,%f,%f,%f,%g\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net));
        fflush( train_csv_file);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if( get_current_batch(net)%10 == 0){
            // Reduced channels
            //image trth = float_to_image(net->w/div, net->h/div, 12, train.y.vals[net->batch*(net->subdivisions-1)]);
            image trth = float_to_image(net->w/div, net->h/div, 6, train.y.vals[net->batch*(net->subdivisions-1)]);
           // image tr = collapse_birds_layers( trth, 1);
           // save_image_16( tr, "truth");
            image im = collapse_image_layers( float_to_image(net->w, net->h, 9, train.X.vals[net->batch*(net->subdivisions-1)]), 1);
            save_image(im, "input");
           // free_image( im);
           // image pr = collapse_birds_layers(pred, 1);
           // save_image_16( pr, "pred");
           // image dist = image_distance( tr, pr);
           // save_image_16(dist, "dist");
           // image diff = image_diff( tr, pr);
           // save_image_16(diff, "diff");
            //------------------------
            //for debug:
            // Reduced channels
            //image spc = make_image( trth.w*3+6, trth.h*12+33, 3);
            image spc = make_image( trth.w*3+6, trth.h*6+15, 3);
            fill_image( spc, 1.);
            image tmp = make_empty_image(0,0,0);
            image masked_pred = make_image( pred.w, pred.h, 2);
            int ind_spc;
            // Reduced channels
            //for( ind_spc = 0; ind_spc <12; ind_spc++)
            for( ind_spc = 0; ind_spc <6; ind_spc++)
            {
                image t = get_image_layer( trth, ind_spc);
                tmp = bird_to_rgb( t, ind_spc);
                free_image( t);
                place_image( tmp, trth.w, trth.h,  0, ind_spc*(trth.h+3), spc);
                free_image( tmp);
            }
            image msk = get_image_layer( pred, 0);
            tmp = bird_to_rgb( msk, 0);
            place_image( tmp, pred.w, pred.h,  2*(pred.w+3), 0, spc);
            free_image( tmp);
            threshold( &msk, 0.15);
            tmp = bird_to_rgb( msk, 0);
            place_image( tmp, pred.w, pred.h,  pred.w+3, 0, spc);
            free_image( tmp);
            // Reduced channels
            //for( ind_spc = 1; ind_spc <12; ind_spc++)
            for( ind_spc = 1; ind_spc <6; ind_spc++)
            {
                image t = get_image_layer( pred, ind_spc);
                tmp = bird_to_rgb( t, ind_spc);
                //if( ind_spc != 5)//depth
                //    normalize_image(tmp);
                place_image( tmp, pred.w, pred.h, 2*(pred.w+3), ind_spc*(pred.h+3), spc);
                free_image( tmp);
                mul_cpu( pred.w*pred.h, msk.data, 1, t.data, 1);
                tmp = bird_to_rgb( t, ind_spc);
                if( ind_spc != 5)//depth
                    normalize_image(tmp);
                place_image( tmp, pred.w, pred.h,  pred.w+3, ind_spc*(pred.h+3), spc);
                if( ind_spc ==3 || ind_spc == 4)
                  memcpy( masked_pred.data + (ind_spc-3)*pred.w*pred.h, t.data, t.h*t.w*sizeof( float));
            //    free_image( t);
                free_image( tmp);
            }
            image gt_vote = hough_vote( trth, 3, 4, msk, 0, 0);
            normalize_image( gt_vote);
            //threshold( &gt_vote, 0.01);
            save_image_16( gt_vote, "gtvote");
            image p_vote = hough_vote( masked_pred, 0, 1, msk, 1,0 );
            normalize_image( p_vote);
            threshold( &p_vote, 0.03);
            save_image_16( p_vote, "pvote");
            save_image( msk, "birds");
            char f[128];
            sprintf( f, "./fatBranch/%d", get_current_batch(net)/10);
            save_image(spc, f);
            save_image(spc, "result");
            image diff = make_image(trth.w, trth.h, 1);
            for(i = 0; i < trth.h*trth.w; ++i){
                diff.data[i] = fabs(fabs(trth.data[3*trth.w*trth.h+i])-fabs(trth.data[4*trth.w*trth.h+i]));
            }
            normalize_image( diff);
            save_image_16( diff, "gt_diff");
            for(i = 0; i < trth.h*trth.w; ++i){
                diff.data[i] = fabs(trth.data[3*trth.w*trth.h+i])+fabs(trth.data[4*trth.w*trth.h+i]);
            }
            normalize_image( diff);
            save_image_16( diff, "gt_sum");
            for(i = 0; i < trth.h*trth.w; ++i){
                diff.data[i] = fabs(fabs(masked_pred.data[i])-fabs(masked_pred.data[masked_pred.w*masked_pred.h+i]));
            }
            mul_cpu( diff.w*diff.h, msk.data, 1, diff.data, 1);
            normalize_image( diff);
            save_image_16( diff, "p_diff");
            for(i = 0; i < trth.h*trth.w; ++i){
                diff.data[i] = fabs(masked_pred.data[i])+fabs(masked_pred.data[masked_pred.w*masked_pred.h+i]);
            }
            mul_cpu( diff.w*diff.h, msk.data, 1, diff.data, 1);
            normalize_image( diff);
            save_image_16( diff, "p_sum");
            free_image(diff);
            //-------------------------------


           // free_image( tr);
           // free_image( pr);
           // free_image( diff);
           // free_image( dist);
            free_image( spc);
            free_image( msk);
            //free_image( gt_vote);
            //free_image( p_vote);
            //return;
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
  image result = make_image( w*2+3, h*6+15, 3);
  fill_image( result, 1.);
  image im = make_empty_image(0,0,0);
  image sized = make_empty_image( 0,0,0);
  //for( n=8; n < plist->size; ++n)
  n = 9;
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
    float *X = in.data;
    float *predictions = network_predict(net, X);
    printf("Predicted: %f\n", predictions[0]);
    image pred = get_network_image(net);
    image tmp = make_empty_image(0,0,0);
    int ind_spc;
    image msk = get_image_layer( pred, 0);
    tmp = bird_to_rgb( msk, 0);
    place_image( tmp, pred.w, pred.h,  dx, dy, result);
    threshold( &msk, 0.15);
    free_image( tmp);
    tmp = bird_to_rgb( msk, 0);
    place_image( tmp, pred.w, pred.h,  dx+w+3, dy, result);
    free_image( tmp);
    for( ind_spc = 1; ind_spc <6; ind_spc++)
    {
        image t = get_image_layer( pred, ind_spc);
        tmp = bird_to_rgb( t, ind_spc);
        normalize_image(tmp);
        place_image( tmp, pred.w, pred.h, dx, ind_spc*(h+3)+dy, result);
        free_image( tmp);
        mul_cpu( pred.w*pred.h, msk.data, 1, t.data, 1);
        tmp = bird_to_rgb( t, ind_spc);
        if( ind_spc != 5)//depth
            normalize_image(tmp);
        place_image( tmp, pred.w, pred.h,  w+3+dx, ind_spc*(h+3)+dy, result);
        free_image( t);
        free_image( tmp);
    }
    char f[128];
    sprintf( f, "./result/%d", n);
    free_image( msk);
    //show_image(im, "orig", 1);
    //show_image(result, "pred", 0);
    image im = collapse_image_layers( in, 1);
    save_image(im, "in");
    free_image( im);
    char pred_name[256];
    strncpy(input, paths[n], 256);
    sprintf( pred_name, "results/pred_%s", basecfg( input));
    save_image(result, pred_name);
  }
  free_image(result);
}

void predict_spc(char *datafile, char *cfg, char *weights, char *filename, int* gpus, char* prefix)
{
  cuda_set_device(gpus[0]);
  network *net = load_network(cfg, weights, 0);
  set_batch_network(net, 1);

  char buff[256];
  char *input = buff;
  list *plist = get_paths(filename);
  char **paths = (char **)list_to_array(plist);
  int n,m, w = 2048, h = 2048;// w = 1920, h = 1080;
  image in = make_image( net->w, net->h, net->c);
  image result = make_image( w*2+3, h*3+6, 3);
  fill_image( result, 1.);
  image im = make_empty_image(0,0,0);
  image sized = make_empty_image( 0,0,0);
  image final_mask = make_empty_image( w,h,1);
  image masked_pos_x = make_image( w,h,1);
  image masked_pos_y = make_image( w,h,1);
  image masked_pos = make_image( w,h,2);
  for( n=8; n < plist->size; ++n)
  //for( n=17; n < plist->size; ++n)
  {
    fill_image( result, 0.);
    fill_image( masked_pos_x, 0.);
    fill_image( masked_pos_y, 0.);
    int oi ,oj;
    for( oi = 0; oi < 3; ++oi)
    {
      for( oj = 0; oj < 3; oj++)
      {
       int dx, dy;
        switch( oi)
        {
          case 0:
            dx = 0;
            break;

          case 1:
            dx = 512;
            break;

          case 2:
            dx = 1024;
            break;
        }
        switch( oj)
        {
          case 0:
            dy = 0;
            break;

          case 1:
            dy = 512;
            break;

          case 2:
            dy = 1024;
            break;
        }
        for( m = 0; m < 9; ++m)
        //for( m = 0; m < 18; m+=2)
        {
          strncpy(input, paths[n-m], 256);
          im = load_image_color(input, 0, 0);
          sized = crop_image(im, dx, dy, net->w, net->h);
          memcpy( in.data + m*in.h*in.w*3, sized.data, 3*sized.h*sized.w*sizeof( float));
          //memcpy( in.data + (m/2)*in.h*in.w*3, sized.data, 3*sized.h*sized.w*sizeof( float));
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
        merge_images( tmp, pred.w, pred.h,  dx, dy, result);
        free_image( tmp);
        tmp = bird_to_rgb( msk, 0);
        threshold( &msk, 0.15);
        merge_images( tmp, pred.w, pred.h,  dx+w+3, dy, result);
        free_image( tmp);
        threshold( &msk, 0.15);
        for( ind_spc = 1; ind_spc <3; ind_spc++)
        {
            image t = get_image_layer( pred, ind_spc);
            tmp = bird_to_rgb( t, ind_spc);
            merge_images( tmp, pred.w, pred.h, dx, ind_spc*(h+3)+dy, result);
            free_image( tmp);
            mul_cpu( pred.w*pred.h, msk.data, 1, t.data, 1);
            tmp = bird_to_rgb( t, ind_spc);
            normalize_image(tmp);
            merge_images( tmp, pred.w, pred.h,  w+3+dx, ind_spc*(h+3)+dy, result);
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
    //sprintf( pred_name, "mixed/pred_%s", basecfg( input));
    save_image(result, "pred");
    final_mask = crop_image( result, w+3, 0, w, h);
    //threshold( &final_mask, 0.3);
    //sprintf( pred_name, "rook/pred_%s_mask", basecfg( input));
    //save_image(final_mask, pred_name);
    image preds = make_image( w, h, 2);
    image tmp = crop_image( result, 0, h+3, w, h);
    normalize_image( tmp);
    memcpy( preds.data, tmp.data, w*h*sizeof(float));
    free_image( tmp);
    tmp = crop_image( result, 0, 2*h+6, w, h);
    normalize_image( tmp);
    memcpy( preds.data+w*h, tmp.data, w*h*sizeof( float));
    free_image( tmp);
    image spcs = bird_species( preds, 0, 1, final_mask);
    if( !prefix)
      sprintf( pred_name, "mixeds/pred_%s_spc", basecfg( input));
    else
      sprintf( pred_name, "%s/pred_%s_spc", prefix, basecfg( input));
    save_image( spcs, pred_name);
    free_image( spcs);
    free_image( preds);
    free_image( final_mask);
  }
  free_image(result);
}

void maskgn(char *datafile, char *cfg, char* prefix)
{
  list *dlist = get_paths( datafile);
  char** imgs = (char **)list_to_array(dlist);
  image mask = make_image( 2048, 2048, 1);
  int i,j;
  char buff[256];
  char *input = buff;
  for( i = 0; i < dlist->size; ++i)
  {
    strcpy( input, imgs[i]);
    image im = load_image_16( input, 1);
    for( j = 0; j < im.w*im.h*im.c; ++j)
    {
      mask.data[j] += im.data[j];
      if( mask.data[j] > 1.0) 
        mask.data[j] = 1.0;
    }
    free_image( im);
  }
  sprintf( input, "%s/sum", prefix);
  save_image_16( mask, input);
  free_image( mask);
}

void maskgt(char *datafile, char *cfg, char* prefix)
{
  list *dlist = get_paths( datafile);
  char** imgs = (char **)list_to_array(dlist);
  image mask = load_image( cfg, 0, 0, 1);
  int i;
  for( i = 0; i < dlist->size; ++i)
  {
    char buff[256];
    char *input = buff;
    strcpy( input, imgs[i]);
    image im = load_image_16( input, 1);
    mul_cpu( im.w*im.h, mask.data, 1, im.data, 1);
    threshold( &im, 0.3);
    sprintf( input, "%s/%s", prefix, basecfg( input));
    save_image_16( im, input);
    free_image( im);
  }
  free_image( mask);
}

void segment_birds(char *datafile, char *cfg, char *weights, char *filename, int* gpus, char* prefix)
{
  cuda_set_device(gpus[0]);
  network *net = load_network(cfg, weights, 0);
  set_batch_network(net, 1);

  list *dlist = get_paths( filename);
  char** dirs = (char **)list_to_array(dlist);
  int d,n,m, w = 2048, h = 2048;// w = 1920, h = 1080;
  for( d = 0; d < dlist->size; ++d)
  {
    char buff[256];
    char *input = buff;
    list *plist = get_paths(dirs[d]);
    char **paths = (char **)list_to_array(plist);
    image other_spc = make_image( w, h, 1);
    image in = make_image( net->w, net->h, net->c);
    image result = make_image( w, h, 1);
    int num_frames = 9;
    image im[ num_frames];
    for( m = 0; m < num_frames; ++m)
      im[m] = make_empty_image(0,0,0);
    image sized = make_empty_image( 0,0,0);
    for( n=8; n < plist->size; ++n)
    //for( n=17; n < plist->size; ++n)
    {
      fill_image( result, 0.);
      int oi ,oj;
      for( m = 0; m < num_frames; ++m)
      {
        strncpy(input, paths[n-m], 256);
        im[m] = load_image_color(input, 0, 0);
      }
      for( oi = 0; oi < 3; ++oi)
      {
        for( oj = 0; oj < 3; oj++)
        {
         int dx, dy;
          switch( oi)
          {
            case 0:
              dx = 0;
              break;

            case 1:
              dx = 512;
              break;

            case 2:
              dx = 1024;
              break;
          }
          switch( oj)
          {
            case 0:
              dy = 0;
              break;

            case 1:
              dy = 512;
              break;

            case 2:
              dy = 1024;
              break;
          }
          for( m = 0; m < num_frames; ++m)
          //for( m = 0; m < 18; m+=2)
          {
            sized = crop_image(im[m], dx, dy, net->w, net->h);
            memcpy( in.data + m*in.h*in.w*3, sized.data, 3*sized.h*sized.w*sizeof( float));
            //memcpy( in.data + (m/2)*in.h*in.w*3, sized.data, 3*sized.h*sized.w*sizeof( float));
            free_image( sized);
          }
          float *X = in.data;
          network_predict(net, X);
          image pred = get_network_image(net);
          image tmp = make_empty_image(0,0,0);
          int ind_spc;
          image msk = get_image_layer( pred, 0);
          threshold( &msk, 0.045);
          merge_images( msk, pred.w, pred.h,  dx, dy, result);
          free_image(tmp);
          free_image(msk);
        }
      }
      char pred_name[256];
      char other_name[256];
      strncpy(input, paths[n], 256);
      char path[256];
      sprintf( path, "%s/jd", basedir( input));
      mkdir( path, 0777);
      sprintf( path, "%s/rk", basedir( input));
      mkdir( path, 0777);
      if( strcmp(prefix, "jd") == 0)
      {
        sprintf( pred_name, "%s/jd/%s_jd", basedir( input), basecfg( input));
        sprintf( other_name, "%s/rk/%s_rk", basedir( input), basecfg( input));
      }
      else
      {
        sprintf( pred_name, "%s/rk/%s_rk", basedir( input), basecfg( input));
        sprintf( other_name, "%s/jd/%s_jd", basedir( input), basecfg( input));
      }
      save_image_16( result, pred_name);
      save_image_16( other_spc, other_name);
      for( m = 0; m < num_frames; ++m)
        free_image( im[m]);

    }
    free_image(result);
    free_image(other_spc);
    free_image( in);
    free_ptrs((void**)paths, plist->size);
  }
}

void seq_crop_predict_segmenter(char *datafile, char *cfg, char *weights, char *filename, int* gpus)
{
  cuda_set_device(gpus[0]);
  network *net = load_network(cfg, weights, 0);
  set_batch_network(net, 1);

  char buff[256];
  char *input = buff;
  list *plist = get_paths(filename);
  char **paths = (char **)list_to_array(plist);
  int n,m, w = 2048, h = 2048;// w = 1920, h = 1080;
  image in = make_image( net->w, net->h, net->c);
  image result = make_image( w*2+3, h*6+15, 3);
  fill_image( result, 1.);
  image im = make_empty_image(0,0,0);
  image sized = make_empty_image( 0,0,0);
  image final_mask = make_empty_image( w,h,1);
  image masked_pos_x = make_image( w,h,1);
  image masked_pos_y = make_image( w,h,1);
  image masked_pos = make_image( w,h,2);
  for( n=8; n < plist->size; ++n)
  {
    fill_image( result, 0.);
    fill_image( masked_pos_x, 0.);
    fill_image( masked_pos_y, 0.);
    int oi ,oj;
    for( oi = 0; oi < 3; ++oi)
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
            dx = 544;
            break;

          case 2:
            dx = 1088;
            break;
        }
        if( oj)
          dy = 248;
        else
          dy = 0;
  //      switch( oi)
  //      {
  //        case 0:
  //          dx = 0;
  //          break;

  //        case 1:
  //          dx = 680;
  //          break;

  //        case 2:
  //          dx = 1216;
  //          break;
  //      }
  //      switch( oj)
  //      {
  //        case 0:
  //          dy = 0;
  //          break;

  //        case 1:
  //          dy = 680;
  //          break;

  //        case 2:
  //          dy = 1216;
  //          break;
  //      }
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
        merge_images( tmp, pred.w, pred.h,  dx, dy, result);
        free_image( tmp);
        tmp = bird_to_rgb( msk, 0);
        threshold( &msk, 0.15);
        merge_images( tmp, pred.w, pred.h,  dx+w+3, dy, result);
        free_image( tmp);
        threshold( &msk, 0.15);
        for( ind_spc = 1; ind_spc <6; ind_spc++)
        {
            image t = get_image_layer( pred, ind_spc);
            tmp = bird_to_rgb( t, ind_spc);
            merge_images( tmp, pred.w, pred.h, dx, ind_spc*(h+3)+dy, result);
            free_image( tmp);
            mul_cpu( pred.w*pred.h, msk.data, 1, t.data, 1);
            if( ind_spc ==3 )
              merge_images( t, pred.w, pred.h,  dx, dy, masked_pos_x);
            if( ind_spc ==4 )
              merge_images( t, pred.w, pred.h,  dx, dy, masked_pos_y);
            tmp = bird_to_rgb( t, ind_spc);
            if( ind_spc != 5)//depth
                normalize_image(tmp);
            merge_images( tmp, pred.w, pred.h,  w+3+dx, ind_spc*(h+3)+dy, result);
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
    //sprintf( pred_name, "results/pred_%s", basecfg( input));
    //save_image(result, pred_name);
    final_mask = crop_image( result, w+3, 0, w, h);
    //threshold( &final_mask, 0.3);
    sprintf( pred_name, "results/pred_%s_mask", basecfg( input));
    save_image(final_mask, pred_name);
    memcpy( masked_pos.data, masked_pos_x.data, h*w*sizeof( float));
    memcpy( masked_pos.data+h*w, masked_pos_y.data, h*w*sizeof( float));
    sprintf( pred_name, "results/pred_%s_hough", basecfg( input));
    save_image(result, pred_name);
    image votes = hough_vote( masked_pos, 0, 1, final_mask, 1, pred_name);
    //image temp = bird_to_rgb( masked_pos_x, 3);
    //save_image( temp, "test_x");
    //temp = bird_to_rgb( masked_pos_y, 4);
    //save_image( temp, "test_y");
    //free_image( temp);
    normalize_image(votes);
    threshold( &votes, 0.045);
    sprintf( pred_name, "results/pred_%s_vote", basecfg( input));
    save_image(votes, pred_name);
    image preds = make_image( w, h, 2);
    image tmp = crop_image( result, 0, h+3, w, h);
    normalize_image( tmp);
    memcpy( preds.data, tmp.data, w*h*sizeof(float));
    free_image( tmp);
    tmp = crop_image( result, 0, 2*h+6, w, h);
    normalize_image( tmp);
    memcpy( preds.data+w*h, tmp.data, w*h*sizeof( float));
    free_image( tmp);
    image spcs = bird_species( preds, 0, 1, final_mask);
    sprintf( pred_name, "results/pred_%s_spc", basecfg( input));
    save_image( spcs, pred_name);
    tmp = crop_image( result, 0,5*( h+3), w, h);
    image t = get_image_layer( tmp, 0);
    free_image( tmp);
    mul_cpu( votes.w*votes.h, votes.data, 1, t.data, 1);
    sprintf( pred_name, "results/pred_%s_depth", basecfg( input));
    save_image_16( t, pred_name);
    free_image( t);
    free_image( spcs);
    free_image(votes);
    free_image( preds);
    free_image( final_mask);
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

    clock_t time;
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

    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
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
    if(0==strcmp(argv[2], "test")) predict_segmenter(data, cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_segmenter(data, cfg, weights, gpus, ngpus, clear, display);
    else if(0==strcmp(argv[2], "demo")) demo_segmenter(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "singletest")) seq_predict_segmenter(data, cfg, weights, filename, gpus);
    else if(0==strcmp(argv[2], "batchtest")) seq_crop_predict_segmenter(data, cfg, weights, filename, gpus);
    else if(0==strcmp(argv[2], "spctest")) predict_spc(data, cfg, weights, filename, gpus, prefix);
    else if(0==strcmp(argv[2], "segbirds")) segment_birds(data, cfg, weights, filename, gpus, prefix);
    else if(0==strcmp(argv[2], "maskgt")) maskgt(data, cfg, prefix);
    else if(0==strcmp(argv[2], "maskgn")) maskgn(data, cfg, prefix);
    else if(0==strcmp(argv[2], "visualize")) predict_visualize_single_crop(data, cfg, weights, filename);
}


