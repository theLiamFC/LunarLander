# LunarLander
## Introduction

## User Instructions

## Dependencies

## Image Downloads
Images are sourced from https://pds.lroc.asu.edu/data/

We are using the Wide Angle Camera (WAC) Region of Interst (ROI) Nearside Dawn image collection. 

Make sure wget is installed with brew install wget.

In your desired folder, run the following commands:
wget \
  -r              \
  -l1             \
  -np             \
  -nd             \
  -A "*DAWN_E*_*100M.IMG,*DAWN_E*_*100M.xml,*README.TXT" \
  "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_ROI/WAC_ROI_NEARSIDE_DAWN/"
