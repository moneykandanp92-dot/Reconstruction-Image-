<img width="923" height="468" alt="image" src="https://github.com/user-attachments/assets/c3c518c7-e882-4358-a57d-da69802fe6d5" /># Computer Vision and Diffusion-Based Reconstruction of Structural Damage Images

## Overview
Structural damage assessment plays a crucial role in ensuring the safety, durability, and maintenance of buildings and infrastructure. Traditional inspection methods rely heavily on manual evaluation by experts, which is time-consuming, costly, and often subject to human error and inconsistency. Although recent advancements in computer vision and deep learning have enabled automated detection of structural defects such as cracks and spalling, most existing systems are limited to identifying damaged regions without providing reconstruction or repair-related insights.
This project proposes an integrated AI-based framework for wall damage detection, reconstruction, and repair estimation using deep learning and rule-based techniques. The system begins by preprocessing input images of damaged structures and applying segmentation models such as U-Net to identify and isolate damaged regions at the pixel level. The segmented output is then used to calculate damage percentage through pixel-based analysis, enabling estimation of the affected area.
For reconstruction, multiple deep learning models, including Autoencoder, Denoising Autoencoder, U-Net, Generative Adversarial Networks (GAN), and Pix2Pix are implemented and compared. These models aim to generate visually improved versions of damaged structures, simulating repaired conditions. The performance of these models is evaluated using quantitative metrics such as (MSE) and (PSNR), IoU, Dice Coefficient along with qualitative visual assessment and Confusion Matrix. Among the models, U-Net and GAN-based approaches demonstrate superior reconstruction quality with lower error and higher PSNR values.
In addition to image reconstruction, the system incorporates a practical rule-based estimation module for calculating material requirements and manpower. Based on the damaged area, the system estimates the quantity of cement and sand required, along with the number of workers and time needed for repair. These estimations are derived from real-world survey data collected from construction workers, ensuring practical relevance and reliability.
The proposed system provides an end-to-end solution that combines damage detection, reconstruction, and estimation into a unified pipeline. This approach not only reduces manual effort but also enhances decision-making in construction, maintenance, and infrastructure management. The project demonstrates the potential of integrating computer vision and generative AI techniques with domain knowledge to support intelligent structural repair planning and future smart construction systems.

## Tools Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy

## Methodology
	Auto encoder model
The Autoencoder [4] model was able to produce reconstructed images, but the quality of the output was low, and the images were slightly blurred. This groundwork to more improvements in future with advanced models. “FIGURE 1,” depicts the simplest structure of the system, which is concerned with the image preprocessing and reconstruction with the help of the Autoencoder model. 
Architecture “FIGURE 1,” consists of the following main components:
<img width="1794" height="877" alt="Autoencoder" src="https://github.com/user-attachments/assets/58ae01ee-002a-4813-9c18-0f8fd61739fd" />


FIGURE 1: Architecture of Autoencoder
1. Input Layer: The system receives a wall damage image as input. 
2. Preprocessing Module: The input image is resized to a standard dimension (128 × 128). Pixel values are normalized to improve model performance. 
3. Noise Generation Module: Gaussian noise is added to the image to create a noisy dataset. 
4. Autoencoder Model: Encoder-Compresses the image into a latent representation. Decoder-Reconstructs the image from compressed data
5. Output Module: The system generates the reconstructed image. 
	U-net model
The U-Net model to do image reconstruction and damage segmentation. The process begins with the input image processing such as resizing and normalization. The processed image is feed to the U-Net model which has encoder (feature extraction) and a decoder (image reconstruction) with skip connections to maintain the spatial information.
The model yields two outputs: a reconstructions image and a segmentation mask with the areas of damage. The output of the segmentation is again utilized to compute the damage region with the help of pixel-based analysis [1] [12]. the system measures the model performance based on MSE and PSNR and all outputs are presented with a visualization module.
 <img width="1794" height="877" alt="Autoencoder" src="https://github.com/user-attachments/assets/3580ebd3-5ea7-4c23-9d28-8310de6076d6" />

FIGURE 2: Architecture of U-Net
The Architecture Diagram for U-Net “FIGURE 2,” improves the simple interface that was created by adding more sophisticated functions like better visualization of reconstruction and display of damage segments. The interface is made to be user friendly and interactive to allow users to upload images, run them on the U-Net model and see detailed outputs such as segmented areas of damages.
The user interface will start with an image upload section where the user can choose and upload wall damage images in a normal format (JPG or PNG). After uploading the image, the user is able to start processing with the help of a “Process” button, which will start the entire pipeline with preprocessing, U-Net model, segmentation, and evaluation.
It depicts the original image, Autoencoder result [4], U-Net reconstructed image and the segmentation mask indicating the damaged areas. Such a comparative representation allows users to have a clear idea about the results of reconstruction quality improvement and accuracy of damage detection.
Besides producing images, the UI also has a results section which shows the evaluation measures like Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR). These measures are objective measures of the model performance. The interface will provide all the outputs in a structured and side-by-side format to be easily interpreted.
	Generative Adversarial Networks (GAN) and Pix2Pix model
To realize high-quality and realistic image reconstruction by employing the state of the art deep learning [7], models, like Generative Adversarial Networks (GAN) and Pix2Pix. This enhancement of the visual quality, creating more realistic results and completing the system by means of incorporating reconstruction, segmentation and evaluation modules into a whole pipeline. 
This system starts with an image of a wall damage that is entered into the system, and it undergoes a preprocessing module. At this step, the image is resized, and normalized so that it can be processed in a model. The preprocessed image is then processed through various reconstruction networks, such as Autoencoder [4], U-Net [1], GAN, and Pix2Pix. The input in each of the models is processed separately to produce reconstructed outputs.
The GAN model “FIGURE 3,” is made up of a discriminator and a generator. The generator is used to generate reconstructed images, and the authenticity of the reconstructed images is compared against the original images by the discriminator, enhancing the quality of the output with adversarial learning. Pix2Pix model is another conditional GAN which is used to produce structured and realistic results as a result of paired data.
The U-Net model “FIGURE 2,” in addition to reconstruction, would give a segmentation mask that would indicate areas of damage in the image. This segmentation output is as the input of the damage area calculation module where a pixel analysis is carried out to calculate the percentage of damaged area.
The output visualization module shows all reconstructed images (Autoencoder [4], U-Net [1], GAN, Pix2Pix) and the segmentation mask as well as evaluation results in a formatted manner. This enables users to make easy comparisons of outputs and determine the efficiency of any of the models.
As Shown “FIGURE 3,” Visualization of Architecture is further facilitated to visualize the output of several results at the same time, GAN output and Pix2Pix output. This will give a holistic picture of the system performance and point out improvements in every stage.
The output visualization module shows “FIGURE 4(a),” all reconstructed images (Autoencoder [4], U-Net [1], GAN, Pix2Pix) and the segmentation mask as well as evaluation results in a formatted manner. This enables users to make easy comparisons of outputs and determine the efficiency of any of the models.

 
FIGURE 3: Architecture of (GAN) and Pix2Pix

 
FIGURE 4(a): Model Comparison-Evaluation
As Shown “FIGURE 4(a),” Mean Squared Error (MSE) measures the average squared difference between the original image and reconstructed image. Lower MSE values indicate lower reconstruction error and better image quality. The MSE formula is represented as Equation below:
MSE=1/MN ∑_(i=1)^M▒∑_(j=1)^N▒〖[I(i,j)-K(i,j)〗 ]^2
Where I(i,j) represent the original image pixel, K(i,j) represents the reconstructed image pixel, and M times N represents the image size.
The obtained MSE values are 0.0008 for Autoencoder, 0.0005 for U-Net, 0.0009 for GAN, and 0.0009 for Pix2Pix. Among these models, U-Net achieved the lowest MSE value, indicating lower reconstruction error compared to other models.
(PSNR) is used to evaluate the quality of reconstructed images. Higher PSNR values indicate better reconstruction quality and lower image distortion. The PSNR formula is represented as Equation Below
PSNR=10〖log⁡〗_10 (〖255〗^2/MSE)
Using the obtained MSE values, the PSNR values were calculated for all reconstruction models. The Autoencoder model achieved a PSNR value of 23.89 dB, U-Net achieved 31.37 dB, GAN achieved 27.13 dB, and Pix2Pix achieved the highest PSNR value of 24.08 dB. The higher PSNR value obtained by Pix2Pix indicates improved visual reconstruction quality and reduced distortion in reconstructed images.
Structural Similarity Index Measure (SSIM) evaluates the structural similarity between the original image and reconstructed image. Unlike MSE and PSNR, SSIM 
considers brightness, contrast, and structural information for image quality assessment. The SSIM formula is represented as Equation below
SSIM(x,y=((2μ_x μ_y+C_1)(2σ_xy+C_2 ))/((μ_x^2+μ_y^2+C_1)(σ_x^2+σ_y^2+C_2 ) )
The obtained SSIM values are 0.70 for Autoencoder, 0.91 for U-Net, 0.79 for GAN, and 0.69 for Pix2Pix. The Autoencoder model achieved the highest SSIM value, indicating better structural similarity between the original and reconstructed images.
Intersection over Union (IoU) measures the overlap between original crack region and reconstructed crack region. Higher IoU values indicate better overlap accuracy and improved crack reconstruction capability. The IoU formula is represented as Equation below:
IoU=TP/(TP+FP+FN)
Where:
TP represents True Positive, FP represents False Positive, FN represents False Negative. 
The obtained IoU values are 0.7643 for Autoencoder, 0.8811 for U-Net, 0.8385 for GAN, and 0.7959 for Pix2Pix. Among these models, U-Net achieved the highest IoU value, indicating better overlap accuracy and improved crack region reconstruction performance.
Dice Coefficient measures the similarity between original crack region and reconstructed crack region. Higher Dice values indicate improved crack preservation performance. The Dice Coefficient formula is represented as Equation below:
Dice=2TP/(2TP+FP+FN)
Where:
TP represents True Positive, FP represents False Positive, FN represents False Negative. 
The obtained Dice Coefficient values are 0.8664 for Autoencoder, 0.9368 for U-Net, 0.9121 for GAN, and 0.8863 for Pix2Pix. Among these models, U-Net achieved the highest Dice Coefficient value, demonstrating superior crack structure preservation and reconstruction capability compared to the other evaluated models.
As Shown “FIGURE 4(a),” The overall experimental results demonstrate that the proposed deep learning models successfully reconstructed damaged wall images with improved quality. Pix2Pix achieved better PSNR performance, U-Net achieved lower reconstruction error, and Autoencoder achieved higher structural similarity. These results confirm the effectiveness of deep learning-based reconstruction methods for wall damage restoration and intelligent repair analysis[31].
 
FIGURE 4(b): Model Comparison-Confusion matrix
The confusion matrix “FIGURE 4(b),” evaluate the reconstruction performance of each model by comparing the original crack regions with the reconstructed crack regions. It measures how accurately the model predicts crack and non-crack pixels in the reconstructed image.
In the confusion matrix “FIGURE 4(b),” True Positive (TP) indicates correctly reconstructed crack pixels, and True Negative (TN) indicates correctly reconstructed background pixels. False Positive (FP) represents background pixels incorrectly identified as crack regions, while False Negative (FN) represents actual crack pixels missed during reconstruction. Higher TP and TN values with lower FP and FN values indicate better reconstruction performance.
From the obtained confusion matrices, the U-Net model achieved the highest TP value of 4422 and the lowest FN value of 169, indicating superior crack preservation and accurate reconstruction capability. The Simplified GAN model achieved the highest TN value of 11382 and the lowest FP value of 411, indicating better background reconstruction performance. Compared to the other models, Autoencoder and Denoising Autoencoder produced higher false negative values, indicating that some crack regions were not reconstructed accurately.
As Shown “FIGURE 4(b),” the confusion matrix analysis confirms that the U-Net model achieved the best structural crack reconstruction performance among all evaluated models due to its higher crack detection accuracy and lower reconstruction error.
	Best‑model selection and performance reporting
A significant deliverable of this project is the choice of the most productive model since several models based on deep learning were created and tested to reconstruct wall damage. The models that are taken into consideration are Autoencoder [4], U-Net [1], Generative Adversarial Network (GAN) and Pix2Pix. All the models were evaluated using qualitative and quantitative parameters.
Qualitatively, reconstructed images were visually examined to assess their clarity, structural preservation and realism. The Autoencoder generated simpler outputs with evident blurring whereas the U-Net enhanced structural detailing and segmentation was possible. The GAN model and Pix2Pix model produced much more realistic and visually improved images, which were almost similar to the original undamaged images.
“FIGURE 5,” shows the reconstruction results obtained from different deep learning models for damaged wall image restoration. The original crack image “FIGURE 5 (a),” is first corrupted by adding noise, and the noisy image “FIGURE 5 (b),” is then reconstructed using Autoencoder, U-Net, GAN, and Pix2Pix models. The reconstruction quality of each model was evaluated using PSNR, SSIM, and MSE metrics.
In “FIGURE 5 (c),” the Autoencoder model reconstructed the damaged wall image with improved structural preservation and smoother texture appearance. The higher SSIM value indicates that the Autoencoder preserved structural similarity effectively between the original and reconstructed images. However, slight blurring can still be observed in some reconstructed regions.
“FIGURE 5 (d),” shows the reconstruction output obtained using the U-Net model. U-Net successfully reconstructed the crack regions and preserved edge information in damaged areas. U-Net reduced reconstruction error, the reconstructed image contains moderate texture smoothing and lower structural similarity.
“FIGURE 5 (e),” the GAN model reconstructed the damaged image using adversarial learning techniques. The GAN model generated visually realistic textures in damaged regions and improved overall image continuity. The lower SSIM value indicates reduced structural similarity, while the higher MSE value shows comparatively larger reconstruction error.
“FIGURE 5 (f),” The higher PSNR value indicates improved visual reconstruction quality and reduced image distortion. Pix2Pix demonstrated balanced performance in terms of reconstruction quality, texture preservation, and structural consistency.

 
FIGURE 5: Best‑model selection
Overall, the experimental results show that each model contributes differently to the reconstruction process. Autoencoder achieved higher structural similarity, U-Net achieved lower reconstruction error, GAN generated visually realistic textures, and Pix2Pix achieved superior reconstruction quality with the highest PSNR performance. 
	Rule-Based Material and Manpower Estimation
To improve the practical applicability of the proposed system, a rule-based estimation approach was incorporated along with deep learning wall damage detection [7].
Table.1:  Mason Survey-Based Estimation data

orkers using a structured Google Form survey Table.1. The survey mainly focused on identifying average construction-related values such as cement consumption, sand consumption, manpower productivity, and repair time required for wall damage repair activities. 

The collected survey responses and their corresponding average values are summarized in Table.1. These values are integrated into the estimation module [11], to calculate repair requirements automatically after damage detection.
 

FIGURE 6: Material and Manpower Estimation Process

The “FIGURE 6,” damaged area is initially identified using the segmentation output generated by the U-Net model. The system calculates the total number of damaged pixels and compares them with the total image pixels to determine the damage percentage [12]. The damage percentage is calculated using the following equation:
Damage % = Damage Pixels/Total Pixels * 100 
For the sample test image, the total wall area considered was 7 sq.ft. The total image pixels were 16384, out of which 2984 pixels were identified as damaged pixels. Using the above formula, the calculated damage percentage was found to be 18.21%.
Damage Area = Total Wall Area * Damage Pixels/Total Pixels 
Based on the calculation, the actual damaged wall area was estimated as 1.27 sq.ft.
After obtaining the damaged wall area, the system performs rule-based material estimation calculations. The required cement quantity is estimated using the formula:
Req Cement=Damage Area * Cement Req per Sq.Ft 
Similarly, the required sand quantity is estimated using:
Req Cement=Damage Area *Sand Req per Sq.Ft 
The repair duration is calculated using the average repair time collected from mason survey responses. The repair time estimation formula is given below:
Repair Time =Damage Area *Time Req per Sq.Ft 
To material estimation, manpower estimation is also performed. Based on the collected field data, it was observed that one worker can repair approximately 35 sq.ft of wall damage per day. Using this reference value, the manpower requirement is calculated using the following formula:
Req Manpower =Damage Area / Worker Capacity per Day 
For the sample output, the estimated damage area of 1.27 sq.ft can be repaired by a single worker within the calculated repair duration of 17.27 minutes as shown “FIGURE 6”.
The proposed estimation module [11], improves the real-world usability of the system by combining artificial intelligence–based damage detection with practical construction estimation knowledge obtained from experienced workers. This integration makes the proposed system more suitable for construction maintenance, repair planning, and future smart infrastructure applications.
	Results of Reconstruction Image with Material and Manpower Estimation (Final System)
The suggested “FIGURE 7,” wall damage reconstruction and estimation system has been developed successfully and tested. The project has the following key outcomes: 
 

FIGURE 7: Architecture Diagram for Reconstruction Image with Material and Manpower Estimation (Final System)


## Results
Model successfully reconstructs images with reduced noise.

## Future Work
- Improve model accuracy
- Use GAN for advanced reconstruction
