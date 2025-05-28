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
Images are sourced from the [LROC archive](https://lroc.im-ldi.com/data/). We use the Wide Angle Camera (WAC) Region of Interest (ROI) Farside Dusk collection at a resolution of 256 pixels per degree (256P).

1. Ensure `wget` is installed:  
```shell
brew install wget
```
2. To selectively download 256P IMG files, corresponding 256P xml files, and the README from LROC - in your target folder run:
```shell
wget -r -l1 -np -nd -A "DUSK_E*256P.IMG,DUSK_E*256P.XML,*README.TXT" "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_ROI/WAC_ROI_FARSIDE_DUSK/"
```
