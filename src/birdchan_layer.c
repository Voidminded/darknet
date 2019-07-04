#include "birdchan_layer.h"
#include "activations.h"
#include "cost_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

char *get_birdChan_string(int c)
{
    switch(c){
        case 0:
            return "Birdness";
        case 1:
            return "JackDaw";
        case 2:
            return "Rook";
        case 3:
            return "Dx";
        case 4:
            return "Dy";
        case 5:
            return "Dz";
        case 6:
            return "Qx";
        case 7:
            return "Qy";
        case 8:
            return "Qz";
        case 9:
            return "Qw";
        case 10:
            return "Sin(Wing)";
        case 11:
            return "Cos(Wing)";
    }
    return "Wow! how many channels do you want?";
}

layer make_birdchan_layer(int batch, int w, int h, int n, int* channels, float scale, ACTIVATION activation, COST_TYPE cost)
{
    layer l = {0};
    l.type = BIRDCHAN;

    l.h = h;
    l.w = w;
    l.c = n;
    l.cost_type = cost;
    l.activation = activation;
    l.indexes = channels;
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.batch = batch;
    l.cost = calloc(1, sizeof(float));
    l.outputs = h*w*l.c;
    l.inputs = l.outputs;
    l.truths = l.w*l.h*l.c;
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.loss = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.scales = calloc(l.h*l.w, sizeof(float));
    l.scale = scale;

    l.forward = forward_birdchan_layer;
    l.backward = backward_birdchan_layer;
    l.state = calloc(n, sizeof(float));
#ifdef GPU
    l.forward_gpu = forward_birdchan_layer_gpu;
    l.backward_gpu = backward_birdchan_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
    l.loss_gpu = cuda_make_array(l.loss, batch*l.outputs);
#endif

    int i;
    fprintf(stderr, "Layers: ");
    for( i = 0; i < n; ++i)
        fprintf(stderr, "%s\t",get_birdChan_string( l.indexes[ i]));
    fprintf(stderr, "Activation: %s", get_activation_string( l.activation));
    fprintf(stderr, " Loss: %s", get_cost_string( l.cost_type));
    fprintf(stderr, "\n");
    srand(0);

    return l;
}

void forward_birdchan_layer(const layer l, network net)
{
    //double time = what_time_is_it_now();
    int i,b,k;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    memset(l.loss, 0, l.outputs * l.batch * sizeof(float));
    memset(l.scales, 0, l.w * l.h * sizeof(float));

#ifndef GPU
    activate_array(l.output, l.batch*l.outputs, l.activation);
#endif

    if(!net.truth)
        return;
    for (b = 0; b < l.batch; ++b){
        memcpy( l.scales, net.truth + b*net.outputs, l.w*l.h*sizeof( float));
        // Other predictions
        for(i = 0; i < l.c; ++i){
            for(k = 0; k < l.w*l.h; ++k){
                if( l.scales[k] > 0.5)//mask is bird
                {
                    int index = b*l.outputs + i*l.w*l.h + k;
                    if(l.cost_type == SMOOTH){
                        smooth_l1_cpu(1, l.output + index, net.truth + b*net.outputs + l.indexes[i]*l.w*l.h + k, l.delta + index, l.loss + index);
                    }else if(l.cost_type == L1){
                        l1_cpu(1, l.output + index, net.truth + b*net.outputs + l.indexes[i]*l.w*l.h + k, l.delta + index, l.loss + index);
                    }else{
                        l2_cpu(1, l.output + index, net.truth + b*net.outputs + l.indexes[i]*l.w*l.h + k, l.delta + index, l.loss + index);
                    }
                }
            }
        }
        for( i = 0; i < l.c; ++i)
            l.state[i] += sum_array(l.loss+i*l.w*l.h+b*l.outputs, l.w*l.h);
    }
    if(((*net.seen)/net.batch)%net.subdivisions == 0){
        printf("Loss: %12.6f", sum_array(l.state,l.c)/(net.batch*net.subdivisions));
        for( i = 0; i < l.c; ++i){
            printf("\t %s:  %12.6f", get_birdChan_string( l.indexes[ i]), l.state[i]/(net.batch*net.subdivisions));
            fprintf( net.log, ",%f", l.state[i]/(net.batch*net.subdivisions));
        }
        printf("\n");
        memset(l.state, 0, l.c * sizeof( float));
    }

    *(l.cost) = sum_array(l.loss, l.batch*l.inputs); //pow(mag_array(l.delta, l.outputs * l.batch), 2);
    //printf("took %lf sec\n", what_time_is_it_now() - time);
}

void backward_birdchan_layer(const layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_birdchan_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu , l.batch*l.outputs, l.activation);
    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_birdchan_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    cuda_push_array(l.loss_gpu, l.loss, l.batch*l.outputs);
//    l2_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
//    cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
//    *(l.cost) = sum_array(l.loss, l.batch*l.inputs);
}

void backward_birdchan_layer_gpu(const layer l, network net)
{
//    int b;
//    for (b = 0; b < l.batch; ++b){
//        //if(l.extra) gradient_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, LOGISTIC, l.delta_gpu + b*l.outputs + l.classes*l.w*l.h);
//    }
    axpy_gpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1);
}



#endif

