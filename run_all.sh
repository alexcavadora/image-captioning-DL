echo "installing dependencies"
base_model
pip install -r requirements.txt

echo "generating vocabulary"
python vocabulary.py

echo "testing vocabulary and dataloader"
python dataset.py

echo "training base model"
python train.py --epochs 10

echo "evaluating final model with a subset of images"
python inference.py --image-dir val_images --model-path outputs/base_model/best_model.pth
