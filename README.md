kaggle_asus
===========


My entry to Kaggle's ASUS Malfunctional Components Prediction - http://www.kaggle.com/c/pakdd-cup-2014.
Top 25% entry.

I wrote a blog post explaining my solution in more detail:

http://blog.alexparij.com/kaggle-asus-failure-survival-analysis.html




The best model was simple survival analysis with some blending of linear regression. I used the excellent Lifelines package https://github.com/CamDavidsonPilon/lifelines for survival analysis. Got into top 25%

lin_comb_survival.py

Also Aalen’s Additive model with sale season as covariates (should have used current seasons not the sale !).The result was ok but too slow

aaf_surival.py

VAR and ARMA time series analysis, didn't manage to make it work well. I used python's statsmodels

arma.py
var.py


