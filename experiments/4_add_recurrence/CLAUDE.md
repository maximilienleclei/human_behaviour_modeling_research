#### Experiment 4:

We expand on @experiments/3_cl_info_dl_vs_ga/

We turn all networks to recurrent networks while maintaining the `[input, 50, output]` structure (the middle layer is the recurrent layer). however it acts like a reservoir, meaning the number of trainable parameters stay the same from experiment 3.

We also add networks of dynamic complexity from @common/dynamic_net.
