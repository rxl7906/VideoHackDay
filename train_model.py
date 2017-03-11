from clarifai.rest import ClarifaiApp

app = ClarifaiApp("3ACWmpsuzmwEvLLBcfRgoUlq2xQPPwx5z-G0pXvx", "1bO9L3FuIygGD4JAajvC8RknUVvnndK3BaALoshg")

# fetch positive images and add to concepts "has_lung_cancer"
app.inputs.create_image_from_filename('./positives/*.jpg', concepts=["has_lung_cancer"], not_concepts=["no_lung_cancer"])
# fetch negative images and add to concepts "no_lung_cancer"
app.inputs.create_image_from_filename('./negatives/*.jpg', concepts=["no_lung_cancer"], not_concepts=["has_lung_cancer"])
# create model with concepts "has_lung_cancer", "no_lung_cancer"
model = app.models.create(model_id="cancer", concepts=["has_lung_cancer", "no_lung_cancer"])
# train model
model = model.train()

# predict with samples