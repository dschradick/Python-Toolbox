########## VISUALISIERUNG - BOKEH - SERVER
# Erlaubt Charts mit dynamischen Daten

#### Grundform
from bokeh.io import curdoc
# => gibt eine aktuelle Seite mit allen Plots und Layouts die erzeugt wurden
# 1. Plots erzeugen
# 2. Callbacks hinzfügen
# 3. Plots und Widgets in Layouts
curdoc().add_root(figure/layout)
# Im Terminal: bokeh server --show myapp.py
# oder directory style apps bokeh server --show  myappdir/


#### Slider
from bokeh.models import Slider
from bokeh.layouts import widgetbox

slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)
slider2 = Slider(title='slider2', start=10, end=100, step=1, value=20)
layout = widgetbox(slider1,slider2)
curdoc().add_root(layout)

## Slider verbinden
from bokeh.models import ColumnDataSource, Slider
from bokeh.layouts import row,column
from bokeh.plotting import figure
from bokeh.io import curdoc
from numpy.random import random,normal,lognormal

N = 300
source = ColumnDataSource(data={'x':random(N), 'y':random(N)}) # CDS wird dann verwendet, wenn man Daten dynamisch updaten will

#### Plots und Widegets
p = figure()
_ = p.circle(x='x',y='y',source=source)
slider = Slider(start=100,end=1000,value=N,step=10,title='Number of Points')

# Callbacks zum widget hinzufügen
# immer gleich: attr=attribut, welches sich ändert; old = alter Wert, new = neuer Wert
def callback(attr,old,new):
    N = slider.value
    source.data={'x':random(N),'y':random(N)}  # => updated automatisch den plot
slider.on_change('value',callback)

#### Arrange Plots und Widgets in Layouts
# layout = column(widgetbox(slider),p)
layout = column(slider,p)
curdoc().add_root(layout)


### Dropdowns
from bokeh.models import ColumnDataSource, Select
# => CDS wird dann verwendet, wenn man Daten dynamisch updaten will

N = 1000
source = ColumnDataSource(data={'x':random(N), 'y':random(N)})
p = figure()
_ = p.circle(x='x',y='y',source=source)
menu = Select(options=['uniform','normal','lognormal'],value='uniform',title='Distribution')

def callback(attr,old,new):
    if menu.value == 'uniform':
            f = random
    elif menu.value == 'normal':
        f = normal
    else:
        f = lognormal
    source.data = {'x': f(size=n), 'y': f(size=N)}
menu.on_change('value',callback)
layout = column(menu,p)
curdoc().add_root(layout)


### Buttons
from bokeh.models import Button

button = Button(label='press me')
def update():
    # do something
button.on_click(update)

### weitere
toggle = Toggle(label='On/Off',button_type='success'))
checkbox = CheckboxGroup(labels=['foo','bar','baz'])
radio = RadioGroup(labels=['2000','2010','2020'])
def callback(active):
    # ...
    # active besagt, welcher button active ist
