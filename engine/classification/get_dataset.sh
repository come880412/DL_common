kaggle competitions download -c dogs-vs-cats

mkdir ../dataset

unzip dogs-vs-cats.zip -d ../dataset
rm dogs-vs-cats.zip

unzip ../dataset/train.zip -d ../dataset
unzip ../dataset/test1.zip -d ../dataset
rm ../dataset/train.zip ../dataset/test1.zip