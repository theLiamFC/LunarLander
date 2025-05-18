# LunarLander

A simulated lunar landing focusing on the implementation of vision based navigation and state estimation techniques.

## Table of Contents
- [Introduction](#introduction)
- [User Instructions](#user-instructions)
- [Dependencies](#dependencies)
- [Image Downloads](#image-downloads)
- [Example Usage](#example-usage)

## Introduction
LunarLander...

## User Instructions
1. **Clone the repository**
```shell
git clone https://github.com/yourusername/LunarLander.git
cd LunarLander
```
3. **Prepare data**  
Download the required WAC ROI images (see [Image Downloads](#image-downloads)).
4. **Run the simulation**  

## Dependencies
- Python 3.7 or higher  
- numpy  
- rasterio  
- affine  
- Pillow  
- wget (for data download)

## Image Downloads
Images are sourced from the [LROC PDS archive](https://pds.lroc.asu.edu/data/). We use the Wide Angle Camera (WAC) Region of Interest (ROI) Nearside Dawn collection.

1. Ensure `wget` is installed:  
```shell
brew install wget
```
2. In your target folder, run to selectively download 100M IMG files, corresponding 100M xml files, and the folder README:
```shell
wget -r -l1 -np -nd
-A "DAWN_E*100M.IMG,DAWN_E*100M.xml,*README.TXT"
"https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_ROI/WAC_ROI_NEARSIDE_DAWN/"
```
