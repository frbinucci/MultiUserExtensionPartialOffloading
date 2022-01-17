@echo off
Rem
set list=1E8
(for %%a in (%list%) do ( 
   python main.py --out ./GOOD_CHANNEL_OFFLOADING/ACCURACY_95 --V %%a
))