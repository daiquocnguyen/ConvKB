<p align="center">
	<img src="https://github.com/daiquocnguyen/ConvKB/blob/master/convkb_logo.png">
</p>

# A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FConvKB%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/ConvKB"><a href="https://github.com/daiquocnguyen/ConvKB/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/ConvKB"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/ConvKB">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/ConvKB">
<a href="https://github.com/daiquocnguyen/ConvKB/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/ConvKB"></a>
<a href="https://github.com/daiquocnguyen/ConvKB/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/ConvKB"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/ConvKB">

This program provides the implementation of the CNN-based model ConvKB for knowledge graph embeddings as described in [the paper](http://www.aclweb.org/anthology/N18-2053). ConvKB uses a convolution layer with different filters of the same `1 × 3` shape and then concatenates output feature maps into a single vector which is multiplied by a weight vector to produce a score for the given triple.

<p align="center"> 
<img src="https://github.com/daiquocnguyen/ConvKB/blob/master/model.png" width="344" height="400">
</p>

## Usage

### News

- June 13: Update Pytorch (1.5.0) implementation. The ConvKB Pytorch implementation, which is based on the OpenKE framework, is to deal with [the issue #5](https://github.com/daiquocnguyen/ConvKB/issues/5#issuecomment-634568519).

- May 30: The Tensorflow implementation was completed approximately three years ago, and now it is out-of-date. I will release the Pytorch implementation soon.

### Requirements

- Python 3.6
- Pytorch 1.5.0 or Tensorflow 1.6 

### Training

Regarding the Pytorch implementation:

	$ python train_ConvKB.py --dataset WN18RR --hidden_size 50 --num_of_filters 64 --neg_num 10 --valid_step 50 --nbatches 100 --num_epochs 300 --learning_rate 0.01 --lmbda 0.2 --model_name WN18RR_lda-0.2_nneg-10_nfilters-64_lr-0.01 --mode train
	
	$ python train_ConvKB.py --dataset FB15K237 --hidden_size 100 --num_of_filters 128 --neg_num 10 --valid_step 50 --nbatches 100 --num_epochs 300 --learning_rate 0.01 --lmbda 0.1 --model_name FB15K237_lda-0.1_nneg-10_nfilters-128_lr-0.01 --mode train
	
|Dataset	|	Standard setting (MR-MRR-Hits@10)	|	[Using type constraints](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)) (MR-MRR-Hits@10)	|
|:-:		|:-:	|:-:  |
|WN18RR 	|2741 - 0.220 - 50.8	|1377 - 0.259 - 54.9	|
|FB15K-237	|196 - 0.302 - 48.3	|105 - 0.338 - 52.7	|

## Cite

Please cite the paper whenever ConvKB is used to produce published results or incorporated into other software:

	@inproceedings{Nguyen2018,
	  author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dat Quoc Nguyen and Dinh Phung},
	  title={{A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network}},
	  booktitle={Proceedings of the 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
	  pages={327--333},
	  year={2018}
	}
		
## License

Please cite the paper whenever ConvKB is used to produce published results or incorporated into other software. I would highly appreciate to have your bug reports, comments and suggestions about ConvKB. As a free open-source implementation, ConvKB is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 

ConvKB  is licensed under the Apache License 2.0.
