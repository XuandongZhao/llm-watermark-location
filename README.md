## Efficiently Identifying Watermarked Segments in Mixed-Source Texts

📄 **Accepted to ACL 2025**

This repository contains the code for the paper [Efficiently Identifying Watermarked Segments in Mixed-Source Texts](https://arxiv.org/abs/2410.03600).

If you find this repository useful, please cite our paper:

```
@article{zhao2024efficiently,
  title={Efficiently Identifying Watermarked Segments in Mixed-Source Texts},
  author={Zhao, Xuandong and Liao, Chenwen and Wang, Yu-Xiang and Li, Lei},
  journal={arXiv preprint arXiv:2410.03600},
  year={2024}
}
```

#### Compile and Run

First in ./cpp_src, compile the c++ code for aligator.cpp

```cmd
Linux:
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` aligator.cpp -o aligator`python3-config --extension-suffix`
Mac:
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` aligator.cpp -o aligator`python3-config --extension-suffix`

```

Part 1 Geometry Cover Detection:

```cmd
cd scripts
bash run_Geometry_Cover_detection.sh
```

Part 2 AOL:

```
cd scripts
bash run_AOL.sh
```



#### Acknowledgement

We are building on the following works:

[Unigram-Watermark](https://github.com/XuandongZhao/Unigram-Watermark)

[three_bricks](https://github.com/facebookresearch/three_bricks)