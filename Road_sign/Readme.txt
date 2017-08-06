Features : 
	['DetectedCamera', 'AngleOfSign', 'SignAspectRatio', 'SignHeight', 'Target']  I have used these features for classification task. After applying Machine learning classifier(Xgboost)  I have found that 'SignWidth' is not important feature. so I removed it.

Machine learning Algorithm:()
	I have used XGBClassifier() for classification task. After Tuning it's parameter found that learning_rate = 0.2 gives good result.
	I hava also plot learning curves for n_estimators , max_depth , learning_rate.

Files Information :
	1.) train.py - All preprocessing and Training steps can be found here.
	2.) Hyperparameter_Tuning.py - All code to tune differnet parameter.
	
	Other information of parameter tunning can be found in these files- 
		a.) learning_rate.txt
		b.) max_depth.txt
		c.) n_estimators.txt
		d.) feature_importance.png
