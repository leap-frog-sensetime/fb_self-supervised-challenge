# fb_self-supervised-challenge

### this project is mainly created for facebook self-supervised learning challenge at ICCV19.

Please check the challenge detail from [challenge website](https://sites.google.com/view/fb-ssl-challenge-iccv19/home#h.p_Yhrh_WnhW4Hh)

For the details of our method, you should check the paper which would be public soon.

The base process of our project is :

0) training your own self-supervised model, which we don't show it here.

1) extract feature using pre-training model, from various datasets like VOC07

2) using these feature representations to test downstream tasks like training a linear-SVM for classification tasks. You could using code in this project to test low-shot and VOC07-classification.

details are explained below.

All the pretrained models and extracted features could be download **[here]**(https://pan.baidu.com/s/1FHSSZCEvTU7llk_EvKBEpw&shfl=shareset), extracting code is: **6i32**

### extracting features from pre-training model.

we use [deeperCluster](https://research.fb.com/publications/unsupervised-pre-training-of-image-features-on-non-curated-data/) as our baseline. And fine-tune it on ImageNet-1k. You could download our fine-tuned model from the link at the beginning.

To extract feature representations using pre-training model, you could use **./feature_extractor/extract_feature.sh**

***to extract features successfully, please set the following parameters:***

1) --data_path: dataset path locally, e.g. '/mnt/lustre/user/data/VOCdevkit/VOC2007'
2) --pretrained: pretrained model deeperCluster 

you could also download our extracted features for our best result from the link at the beginning.

### voc07 challenge track

You could train your own SVM using extracted feature representations, and you could also test trained-SVM on VOC07 testSplit using resources provided by us. The trained-SVM model of our best resutlt is **./voc07/trained_svm.zip** you could uncompress it and test it.

To train and test your svm from scratch, you could follow:

1) ./voc07/train_svm.sh
2) ./voc07/test_svm.sh

### low-shot track

Training and testing instructions of this track is similar to voc07 track. You could download trained SVM model and sampled targets of voc07 trainval split for low-shot track from the link at the beginning.

To train and test from scratch, you could follow:

1) ./low-shot/create_voc_low_shot_samples.sh  // generate low-shot samples
2) ./low-shot/train_svm_low_shot_all.sh       // train low-shot features
3) ./low-shot/aggre_svm_low_shot.sh           // test and aggregate testing results.

### places05 track

To train from scratch, you could use:

./places/eval_linear_places205.sh

Use our trained places model(you could download trained places model from the link at the beginning.

./places/eval_linear_places205_save_json.sh
