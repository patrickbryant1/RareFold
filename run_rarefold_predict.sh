#!/usr/bin/env bash

################################################################################
# TJN25 Contributed Header
#
# This section adds robust command-line argument parsing and path handling
# to make the script a reusable tool.
################################################################################

# Set BASE as the install directory of the script
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE
done
BASE=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

# Set some default values
NUM_REC_DEFAULT=3
OUTDIR_DEFAULT="./output"
PARAMS_DEFAULT="$BASE/data/params/params20000.npy"


usage() {
  cat <<EOF
Usage: $(basename "$0") -i <ID> [OPTIONS]

Runs the RareFold prediction workflow with robust path handling.

Required Arguments:
  -i, --id <string>         A unique identifier for this prediction run.

Optional Arguments:
  -d, --datadir <path>      Directory containing the input FASTA file.
                            (Default: current directory)
  -o, --outdir <path>       Directory where all output will be saved.
                            (Default: ${OUTDIR_DEFAULT})
  -n, --num-recycles <int>  Number of recycles for the model.
                            (Default: ${NUM_REC_DEFAULT})
  -p, --params <file>       Path to the model parameters (.npy) file.
                            (Default: auto-detected within script directory)
  --fasta <file>            Override the path to the input FASTA file. By default,
                            it is assumed to be located at <datadir>/<id>.fasta.
  --msa <file>              Provide a pre-computed MSA file (.a3m). This will
                            skip the hhblits execution step.
  --hhblitsdb <path>        Override the path to the HHblits database.
                            (Default: auto-detected within script directory)
  -h, --help                Display this help message and exit.

EOF
  exit 1
}

# Variables set by CLI args
# Required:
ID=""
# Optional with defaults:
DATADIR="."
NUM_REC=$NUM_REC_DEFAULT
PARAMS=$PARAMS_DEFAULT
OUTDIR=$OUTDIR_DEFAULT
# Optional overrides for constructed paths:
FASTA_OVERRIDE=""
MSA_OVERRIDE=""
HHBLITSDB_OVERRIDE=""

# Get command line args
while [ $# -gt 0 ]; do
  case "$1" in
    -i|--id)
      if [ -z "$2" ] || [[ "$2" == -* ]]; then echo "Error: $1 requires a value." >&2; exit 1; fi
      ID="$2"
      shift 2
      ;;
    -d|--datadir)
      if [ -z "$2" ] || [[ "$2" == -* ]]; then echo "Error: $1 requires a value." >&2; exit 1; fi
      DATADIR="$2"
      shift 2
      ;;
    -n|--num-recycles)
      if [ -z "$2" ] || [[ "$2" == -* ]]; then echo "Error: $1 requires a value." >&2; exit 1; fi
      NUM_REC="$2"
      shift 2
      ;;
    -p|--params)
      if [ -z "$2" ] || [[ "$2" == -* ]]; then echo "Error: $1 requires a value." >&2; exit 1; fi
      PARAMS="$2"
      shift 2
      ;;
    -o|--outdir)
      if [ -z "$2" ] || [[ "$2" == -* ]]; then echo "Error: $1 requires a value." >&2; exit 1; fi
      OUTDIR="$2"
      shift 2
      ;;
    --fasta)
      if [ -z "$2" ] || [[ "$2" == -* ]]; then echo "Error: $1 requires a value." >&2; exit 1; fi
      FASTA_OVERRIDE="$2"
      shift 2
      ;;
    --msa)
      if [ -z "$2" ] || [[ "$2" == -* ]]; then echo "Error: $1 requires a value." >&2; exit 1; fi
      MSA_OVERRIDE="$2"
      shift 2
      ;;
    --hhblitsdb)
      if [ -z "$2" ] || [[ "$2" == -* ]]; then echo "Error: $1 requires a value." >&2; exit 1; fi
      HHBLITSDB_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      usage
      ;;
  esac
done


# Validate required arguments
if [ -z "$ID" ]; then
  echo "Error: Missing required argument: -i/--id" >&2
fi

# Set up remaining variables
mkdir -p "$OUTDIR"

if [ -n "$FASTA_OVERRIDE" ]; then
  FASTAFILE="$FASTA_OVERRIDE"
else
  FASTAFILE="$DATADIR/$ID.fasta"
fi

if [ -n "$MSA_OVERRIDE" ]; then
  MSA_A3M="$MSA_OVERRIDE"
else
  MSA_A3M="$OUTDIR/$ID.a3m"
fi

if [ -n "$HHBLITSDB_OVERRIDE" ]; then
  HHBLITSDB="$HHBLITSDB_OVERRIDE"
else
  HHBLITSDB="$BASE/data/uniclust30_2018_08/uniclust30_2018_08"
fi

# Ensure the final input FASTA file exists before proceeding
if [ ! -f "$FASTAFILE" ]; then
    echo "Error: Input FASTA file not found at: $FASTAFILE" >&2
    exit 1
fi

# Construct paths for intermediate and output files
# The sequence has to be defined with the noncanonical amino acids (NCAAs) in
# threeletter code separated by hyphens, and the regular in oneletter e.g.
# >1FU0_A
# MEKKEFHIVAETGIHARPATLLVQTASKFNSDINLEYKGKSVNLK-SEP-IMGVMSLGVGQGSDVTITVDGADEAEGMAAIVETLQKEGLA
#
# We also need a fasta where the NCAA is "X" for the MSA search
# >1FU0_A
# MEKKEFHIVAETGIHARPATLLVQTASKFNSDINLEYKGKSVNLKXIMGVMSLGVGQGSDVTITVDGADEAEGMAAIVETLQKEGLA
FASTAWITHX="$OUTDIR/${ID}_X.fasta"

MSA_FEATS="$OUTDIR/msa_features.pkl"

# Set the hhblits command. Check if hhblits is in the system PATH first,
# and if not, fall back to the original hardcoded path.
if command -v hhblits &> /dev/null; then
  HHBLITS_CMD="hhblits"
else
  # This is the original path from the developer
  HHBLITS_CMD="$BASE/hh-suite/build/bin/hhblits"
fi

################################################################################
# Original Script Logic (changes to the execution paths for hhblits and predict_sc.py)
#
# The code below is the original workflow. All variables are now defined with
# absolute paths by the header above.
################################################################################


#Write individual fasta files for all unique sequences
if test -f $MSA; then
	echo $MSA exists
else
	 $HHBLITS_CMD -i $FASTAWITHX -d $HHBLITSDB -E 0.001 -all -n 2 -oa3m $MSA
fi


#########Step2: Make MSA features#########
#Here we also use "FASTAWITHX", the NCAA frames and features are mapped
#later in the predict script when the MSA is sampled
MSA_FEATS=$OUTDIR/msa_features.pkl
if test -f $MSA_FEATS; then
	echo $MSA_FEATS exists
else
	python3 $BASE/src/make_msa_seq_feats.py --input_fasta_path $FASTAWITHX \
	--input_msas $MSA --outdir $OUTDIR
fi

#########Step3: Predict#########
#The NCAAs will be saved to chain B in the prediction for easier visualisation
python3 $BASE/src/predict_sc.py --predict_id $ID \
--MSA_feats $MSA_FEATS \
--fasta $FASTAFILE \
--num_recycles $NUM_REC \
--params $PARAMS \
--outdir $OUTDIR
