#############ARGS#############
BASE=. #Change this depending on your local path
ID=1ssc_1afk #Targets - now the same as an example

#############PARAMS FOR DESIGN#############
MAX_RECYCLES=3 #max_recycles - increase if low plDDT
BINDER_LENGTHS="10,11,12,13,14,15"
NITER=1000
RESAMPLE_FREQ=100 #How often to resample the MSA
BATCH_SIZE=2 #The batch size determines how many design threads are run simultaneously
NUM_MSA_CLUSTS=128
PARAMS=$BASE/data/params/params50000.npy
RARE_AAS="MSE,MLY,PTR,SEP,TPO,MLZ,ALY,HIC,HYP,M3L,PFF,MHO" #Specify the threeletter code for the NCAA you want to use for design
CYCLIC=True #Set to False if you want to design linear binders
MAX_WORKERS=30 #Number of CPU instances to use for the threading
OUTDIR=./data/design_test_case/

#######Step2: Make MSA features#######
MSA_FEATS1=$BASE/data/design_test_case/1ssc/msa_features.pkl
MSA_FEATS2=$BASE/data/design_test_case/1afk/msa_features.pkl
MSA_FEATS="$MSA_FEATS1,$MSA_FEATS2"

#######Step3: Design#######
#All metrics will be saved to "metrics.csv" in outdir
#If the run stops for some reason - don't worry - it will continue where it left off
python3 $BASE/src/mc_design_length_var_mt_batch.py --predict_id $ID \
--MSA_feats $MSA_FEATS \
--num_recycles $MAX_RECYCLES \
--binder_lengths $BINDER_LENGTHS \
--num_iterations $NITER \
--resample_every_n $RESAMPLE_FREQ \
--batch_size $BATCH_SIZE \
--params $PARAMS \
--rare_AAs $RARE_AAS \
--cyclic_offset $CYCLIC \
--num_clusters $NUM_MSA_CLUSTS \
--max_workers $MAX_WORKERS \
--outdir $OUTDIR
