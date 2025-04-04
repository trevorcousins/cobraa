#!/bin/bash

# Set variables
BAM_FILE="sorted_input.bam"
REFERENCE_FA="/path/to/reference.fa"
MSMC_TOOLS_DIR="/path/to/msmc-tools"
MAPPABILITY_MASKS_DIR="/path/to/mappability/masks"
MEAN_COV=30  # Replace with your calculated mean coverage

# Process each chromosome
for CHROM in {1..22}; do
  echo "Processing chromosome $CHROM..."
  
  # Generate VCF and BED files
  bcftools mpileup -q 20 -Q 20 -C 50 -r chr$CHROM -f $REFERENCE_FA $BAM_FILE | \
  bcftools call -c -V indels | \
  $MSMC_TOOLS_DIR/bamCaller.py $MEAN_COV chr$CHROM.bed.gz | \
  gzip -c &gt; chr$CHROM.vcf.gz
  
  # Generate MHS file
  python $MSMC_TOOLS_DIR/generate_multihetsep.py \
    --mask=chr$CHROM.bed.gz \
    --mask=$MAPPABILITY_MASKS_DIR/chr$CHROM.bed \
    chr$CHROM.vcf.gz &gt; chr$CHROM.mhs
done

echo "All done!"