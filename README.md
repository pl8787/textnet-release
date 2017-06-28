TextNet
======

TextNet is a deep neural netwrok framework for text matching tasks. 

* [Contributors](https://github.com/pl8787/textnet-release/graphs/contributors)

Quick Start
=====
1. Make a copy of `Makefile.config.example`.

  ``` bash
  cp Makefile.config.example Makefile.config
  ```
  
2. Setting the environment path in `Makefile.config`.

3. Compile it!

  ```
  mkdir bin
  make all -j 16
  ```
  
4. Wirte you own network config file in `json`. 
   You can find examples in [Textnet Model](https://github.com/pl8787/textnet-model).
   
5. Run!
  ```
  ./bin/textnet [model_file]
  ```

Features
=====

* Construct Models: [MODEL_CONFIG](MODEL_CONFIG.md)
* Data Format: [DATA_FORMAT](DATA_FORMAT.md)

Dependences
=====
* mshadow: [Matrix Shadow](https://github.com/dmlc/mshadow)
* jsoncpp: [Json on CPP](https://github.com/open-source-parsers/jsoncpp)
* ZeroMQ: [Zero Message Queue](http://zeromq.org/) [src](https://github.com/zeromq/libzmq)
* d3: [Data-Driven Documents](http://d3js.org/) [For future features]

Version
======
v1.0

Related Projects
=====
* cxxnet: [CXXNET](https://github.com/dmlc/cxxnet)
* Caffe: [Caffe](https://github.com/BVLC/caffe)

Acknowledgements
=====
The following people contributed to the development of the TextNet project：

- **Liang Pang** 
    - Institute of Computing Technolgy, Chinese Academy of Sciences
    - [Google Scholar](https://scholar.google.com/citations?user=1dgQHBkAAAAJ&hl=zh-CN)
- **Sengxian Wan**
    - Institute of Computing Technolgy, Chinese Academy of Sciences
- **Yixing Fan**
    - Institute of Computing Technolgy, Chinese Academy of Sciences
    - [Google Scholar](https://scholar.google.com/citations?user=w5kGcUsAAAAJ&hl=en)
- **Yanyan Lan**
    - Institute of Computing Technolgy, Chinese Academy of Sciences
    - [HomePage](http://www.bigdatalab.ac.cn/~lanyanyan/)
- **Jiafeng Guo**
    - Institute of Computing Technolgy, Chinese Academy of Sciences
    - [HomePage](http://www.bigdatalab.ac.cn/~gjf/)
- **Jun Xu**
    - Institute of Computing Technolgy, Chinese Academy of Sciences
    - [HomePage](http://www.bigdatalab.ac.cn/~junxu/)
- **Xueqi Cheng**
    - Institute of Computing Technolgy, Chinese Academy of Sciences 
    - [HomePage](http://www.bigdatalab.ac.cn/~cxq/)
