# LunarLander
## Introduction

## User Instructions

## Dependencies

## Image Downloads
Images are sourced from https://pds.lroc.asu.edu/data/

We are using Wide Angle Camera (WAC) Region of Interst (ROI) North Summer. 

Make sure wget is installed with brew install wget

In your desired folder, run the following commands:
wget -r \
     -np \
     -nH \
     --cut-dirs=8 \
     -R "index.html*" \
     https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_ROI/WAC_ROI_NORTH_SUMMER/