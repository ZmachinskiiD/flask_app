<h2 align="center"> University1652-Baseline </h2>
##от Божко

##как пользоваться?
### 1. создать базу фичей
первым делом добавьте датасет в папку static/data, важно чтобы сохранялась следующая структура:
static/data/любое_имя/dataset/img1.jpg
static/data/любое_имя/dataset/img2.jpg
...

извлеченные фичи сохранятся в static/data/любое_имя/features.mat. Данная структура обуславливается использованием torhvision.datasets.ImageFolder, который по умолчанию работает сo структурой
root/class/img1.jpg
root/class/img2.jpg
...

и используется соответственно, для классификации изображений.
таким образом /любое_имя/ это /root/ для ImageFolder. Хотя это костыльный подход, я пробовал разные, в том числе унаследовать ImageFolder или родителей и переопределить метод find_classes, как было указано в документации, получалось лишь больше проблем.

далее нужно посчитать фичи: запустите скрипт un_createdb.py. в DataLoader можно изменить batch_size - количество загружаемых за раз картинок и num_workers.

### 2. запустить веб приложение
скачать модели:
[GoogleDrive](https://drive.google.com/open?id=1iES210erZWXptIttY5EBouqgcF5JOBYO) или [OneDrive](https://studentutsedu-my.sharepoint.com/:u:/g/personal/12639605_student_uts_edu_au/EW19pLps66RCuJcMAOtWg5kB6Ux_O-9YKjyg5hP24-yWVQ?e=BZXcdM). после скачивания, поместите модели в ./model/. пример:
./model/three_view_long_share_d0.75_256_s1_google/net_119.pth
./model/three_view_long_share_d0.75_256_s1_google/opts.yaml
укажите в prepare_model имя папки с моделью и опции.
запустить deploy.py

##возникшие проблемы
как указано выше, так как изменить поведение по умолчанию ImageFolder-а не получилось, выбран костыльный подход. он же используется и для того, чтобы считать фичи загружаемых в веб приложении картинок: пусть загрузили картинку img123.jpg. она скопируется в static/temp/img123/img/img123.jpg, лишь затем посчитаются фичи. в связи с тем, что картинку нужно копировать, возникла другая проблема: почему-то нужно передавать абсолютный путь до загруженной картинки, иначе скрипт падает.

также возникла проблема с путями до хранимых в базе картинок: фласк почему-то считает, что они находятся в /ImgSelected/upload/реальный_путь. решить нормальным способом не получилось, поэтому я просто прохожу на две папки назад: '../../' + /ImgSelected/upload/реальный_путь

##датасет
[OneDrive]:
https://1drv.ms/u/s!AgLQOHbBZYGH0myJgpqFzsi2cgcv?e=QrfZHQ
[Backup] GoogleDrive:
https://drive.google.com/file/d/1iVnP4gjw-iHXa0KerZQ1IfIO0i1jADsR/view?usp=sharing
[Backup] Baidu Yun (China):
https://pan.baidu.com/s/1H_wBnWwikKbaBY1pMPjoqQ password: hrqp
[Backup] Mega (NewZealand):
https://mega.nz/file/WJpz1C5a#E7gNKV5b8aOIRtB7ZEYdb45_DpbVNNCpEV0Jh9Sx14s