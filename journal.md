**Quick copy** 

```python
apsisdev@202.53.174.4
```



```html
# <span style="color:lime">```MainDivision```</span>
### <span style="color:indigo">```SubDivision```</span>
<span style="color:darkorange">**UmbrellaTask**</span> 
<span style="color:red">**Subtask**</span>
```

# 

# Fixed Resources:

- [x] Fonts
    - [x] Bangla
    - [x] English
- [x] Signs
- [x] Human Face Images
- [x] Smart Card Template
- [x] National ID Card Template

# dataLib
- [x] data.py
- [x] fix-vocab
- [x] text data
- [x] image data
- [x] segmentation data
  - [x] background
  - [x] rotation
  - [x] perspective transformation
  - [x] placement
  - [x] data points
  - [x] server_gen
  - [x] tfrecords
    - [x] image
    - [x] mask
    - [x] class
    - [x] bbox
  - [x] modeling
  - [x] loss
  - [x] train
  - [ ] inference
- [ ] recognition data
  - [x] location based scene text
  - [ ] encoded and padded labels
  - [ ] data gen


# smart card creation Location: 
* after rec data extraction use this for execution if needed 

```python
text={
        "bn_name"     :   {"location":[327, 185, 777, 245],"font_size":48,"lang":"bn","font":"bold"},
        "en_name"     :   {"location":[327, 270, 777, 318],"font_size":32,"lang":"en","font":"bold"},
        "f_name"      :   {"location":[327, 332, 777, 403],"font_size":48,"lang":"bn","font":"reg"},
        "m_name"      :   {"location":[327, 410, 777, 485],"font_size":48,"lang":"bn","font":"reg"},
        "dob"         :   {"location":[480, 495, 777, 550],"font_size":38,"lang":"en","font":"reg"},
        "nid"         :   {"location":[480, 550, 777, 590],"font_size":42,"lang":"en","font":"bold"}
    }
```