# Create a local folder and cd into it
mkdir WAC_ROI_NORTH_SUMMER
cd WAC_ROI_NORTH_SUMMER

# Recursively download everything in that remote folder
wget -r \
     -np \
     -nH \
     --cut-dirs=8 \
     -R "index.html*" \
     https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_ROI/WAC_ROI_NORTH_SUMMER/
