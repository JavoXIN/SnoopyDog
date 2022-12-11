# SnoopyDog
A project to train these dog breeds from a photo to get more accurate models for recognizing dog breeds with small errors.
> Laptop:
  * Model: HP
  * Device name	DESKTOP-FG13PRQ.
  * Processor	Intel(R) Core(TM) i7-6500U CPU @ 2.50GHz   2.59 GHz.
  * Installed RAM	8.00 GB (7.89 GB usable).
  * Device ID	____________________________________.
  * Product ID	_____________________________.
  * System type	64-bit operating system, x64-based processor.
  * Pen and touch	Pen and touch support with 10 touch points.
> Files:
- [TrainFile](https://github.com/JavoXIN/SnoopyDog/blob/main/dataDog/train) - The FastAdapter is here to simplify creating adapters for RecyclerViews.
- [DataSetFile](https://github.com/JavoXIN/SnoopyDog/blob/main/dataDog/dataset.py) - 데이터
- [Convert_codefile_to_tflite](https://github.com/JavoXIN/SnoopyDog/blob/main/dataDog/convert_to_tflite.py) -COnvert to TFLITE
- [TfliteFile](https://github.com/JavoXIN/SnoopyDog/blob/main/dataDog/b0_tflite) -TFLITE

  >  *트리 에포크 후에 우리는 검증 데이터에서 거의 60% 이상의 정확도를 얻습니다.  즉, 우리가 어떤 식으로든 사진을 찍을 때 첫 번째 예측, 즉 가장 높은 예측을 의미하는 최고의 정확도입니다.  그것은 60%의 경우에 맞습니다.  100, 120가지가 넘는 개 품종이 있고 매우 큰 모델에서 먼저 훈련시키지 않았기 때문에 실제로 꽤 괜찮다고 생각한다면.  그래서 당신은 확실히 그것을 바꿀 수 있고 그것은 매우 쉬울 것입니다. Train File file에서 이 숫자를 b3 등으로 변경한 다음 해당 모델에 해당하는 이미지 크기를 변경하면 고정된 3-4를 사용하는 학습률을 낮출 수도 있습니다.  학습 속도 스케줄러를 사용하여 더 낮은 것을 변경한 다음 데이터 세트를 확장할 수도 있습니다. 모델이 실제로 어려움을 겪는 클래스를 확인한 다음 데이터를 스크랩하는 영리한 방법으로 이를 수행할 수 있습니다.  그 개 품종만을 위해.*

> * 또한 이 데이터 세트의 유효성 검사에서 83% 이상의 정확도를 얻는 더 많은 데이터에 대해 더 큰 모델을 교육한 github 소스 코드에서 사용할 수 있는 모델이 있을 것입니다.  그래서 당신은 또한 그것을 다운로드할 수 있습니다.*
> Photo of Result:
<img src="https://raw.githubusercontent.com/JavoXIN/SnoopyDog/main/resultofDog/SnoopyDogDataTraining.JPG" gravity = "center" width="95%">


