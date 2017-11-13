# 595Project

#Fall Detection and Activity Recognition from RGBDdatasets 
**Jianhang Chen (JianHangChen), Vishveswaran Jothi (Vishwaatpurdue)**
**Goals :**
Develop a new algorithm to recognize different daily activities and to predict when a person will fall performing the activity using 3D videos. Classify whether the performed action is daily activity such as sitting, walking, lying on the mattress or falling while trying to perform those actions using the developed model. Apply the fine tuned model to classify 3D videos with different backgrounds and illumination in real time thus generalizing the model as well as making it as small as possible. The dataset used has RGBD videos collected from single Kinect v2 sensor. Obtain Skeleton data of humans performing various actions.
**Challenges :**
1. Collection of real time dataset other than available ones (approx 88 videos) 
2. Obtain skeleton data of the person on each frame. 
3. Using sequence of skeleton data to represent the feature vector to learn the activity 
4. Smaller network to learn the activity and predict fall (not just to classify fall activity on the video) 
5. Apply the model to predict fall as activity from other datasets that are not used for training (utilizing occlusion of human poses)
**Role of Vishveswaran Jothi** 
1. Generate skeleton data from the videos with time stamps. 
2. Provide feature vector to the network / network architecture. 
3. Obtain more dataset if possible or create own dataset using Kinect v2 for testing. 
4. Convert the developed model into a transfer learning algorithm to learn other activitiesfrom 3D videos 
5. To automate data extraction from videos and make the software useful to implement for various scenarios without human interference.
**Role of Jianhang Chen** 
1. Design of the network 
2. Finalizing the hyper parameters such as learning rate , regularization, learning function etc. 
3. Validating the model with existing models such as Inception and VGG16. 
4. Comparing the accuracy and ROC curve for Inception and VGG16 and designed model.
**Restrictions**
1. If fall prediction accuracy is not satisfiable then try to *predict the activity* using the same model on different datasets.

**References**
1. S. Gasparrini, E. Cippitelli, E. Gambi, S. Spinsante, J. Wahslen, I. Orhan and T. Lindh, “Proposal and Experimental Evaluation of Fall Detection Solution Based on Wearable and Depth Data Fusion”, ICT Innovations 2015, Springer International Publishing, 2016. 99­108, doi:10.1007/978­3­319­25733­4_11. 
