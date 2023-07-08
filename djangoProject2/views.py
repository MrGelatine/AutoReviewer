import math

from django.template.response import TemplateResponse


from keras import backend as K
import tensorflow_addons as tfa
from manage import reloaded_model
import tensorflow as tf

def index(request):
    review = request.POST.get("review", "Undefined")
    prob = None
    if(review != "" and review != "Undefined"):
        #Evaluate
        res = tf.clip_by_value(reloaded_model(tf.constant([review])) * 10, 1, 10).numpy()[0][0]
        prob = {'res': round(min(max(res,1),10),0) }
        #Stop Evaluate
    else:
        prob = {'res': -1}

    return TemplateResponse(request, "index.html", context=prob)

