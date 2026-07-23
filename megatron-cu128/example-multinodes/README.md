# test example for multinode pretrain

- run inside an interative qsub shell
1. Log into an interactive node (select=2)
1. run `./test-multinode.sh`

- run with qsub
1. update value indicated in `submit-multinode.pbs`
1. run `qsub submit-multinode.pbs` where you will get a job id back
    - `qstat -u <userid>` to get the job status
    - `tail -f megatron_multinode.o<jobid>` to track log, because `#PBS -j oe` will join std/err log and write to this file
    - `qdel <jobid>` to cancel the job