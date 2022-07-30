# Criminal Identity Detector

It is my B.Tech final project.

It is a Python Tkinter based desktop application that helps to identify criminal based on eye witness  information. It draws the image of criminal based on user voice input and search the images from the database. It replaces the work of forensic artist and saves the time of searching the  criminal. It does this quickly and accurately using deep learning algorithms.

It will be used by investigation teams because searching the criminal by uncleared image is very time consuming task. It need images of all the people of India which can be possible if this project is hand over to  government. Peoples facial features change at different phases of life so  it need image database creation for one time and will update these image using AI aging algorithm.

## Application Architecture
<img src="https://user-images.githubusercontent.com/37059870/181914027-2780d8f2-b02e-4700-a404-0ba17a948dc1.png" width="450" height="250" />

## Modules
<img src="https://user-images.githubusercontent.com/37059870/181914078-677a98a0-deb1-41d6-b56f-41466fca2e33.png" width="500" height="200" />

## Application Screen Shots

#### Home Page
<img src="https://user-images.githubusercontent.com/37059870/181914098-8d4e3d39-16c9-461e-b517-ed57bd8aa50d.png" width="450" height="300" />

#### Output - Text to Image by StackGAN
<img src="https://user-images.githubusercontent.com/37059870/181914162-e2fd94f4-3d04-4e10-9658-dca27d80bba5.png" width="450" height="300" />

#### Output - Fetched image from database on comparison with generated image by Siamese Neural Network
<img src="https://user-images.githubusercontent.com/37059870/181914230-321a0dff-9a84-4a42-9743-f37effd703d9.png" width="450" height="300" />

#### Output - If image not found in database
<img src="https://user-images.githubusercontent.com/37059870/181914274-52f5f59b-2a8f-4b4c-ac49-6e5b5eca7bf2.png" width="450" height="300" />

## Algorithms
### Stack GAN
It is used to convert text as a input into image as a output.

#### Model Architecture
<img src="https://user-images.githubusercontent.com/37059870/181914289-dbd764d9-3aea-43f2-96a4-2309866c8a28.png" width="450" height="250" />

### Siamese Neural Network
It is used to compare the two images to find similarity in them.

#### Model Architecture
<img src="https://user-images.githubusercontent.com/37059870/181914304-f5811f7c-253a-4227-9a72-90889d56830f.png" width="450" height="250" />

## Technologies
- Google Speech Recognition API
- PyAudio Library
- Auto PY to EXE
