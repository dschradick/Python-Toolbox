########## VISUALISIERUNG - BOKEH - BASICS
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import output_file,show
from bokeh.plotting import figure
from bokeh.sampledata.iris import flowers as df
from bokeh.sampledata.autompg import autompg as auto
from bokeh.sampledata.gapminder import fertility, life_expectancy,population,regions


#### Einfacher Plot
# Weitere Tools:
# box_zoom,reset,pan,lasso_select,hover,box_select
p = figure(x_axis_label='x',y_axis_label='y',tools='wheel_zoom')
_ = p.circle([1,2,3,4,5],[8,6,5,2,4])
output_file('plot.html')
show(p)


#### Verschiedene Plots

## Scatter Plots
# Markers: circle,cross,triangle,asterix,square,inverted_triangle,x
from bokeh.sampledata.autompg import autompg as auto
_ = p.circle([1,2,3,4,5],[8,6,5,2,3]) # Glyphen hinzufügen
_ = p.circle([1,2,3,4,5],[8,6,5,2,4],selection_color='red',nonselection_fill_alpha=0.2,nonselection_fill_color='gray')
#_ = p.diamond(x=[1,2,3,4,5],y=[8,6,5,2,3],size=[50,6,5,2,3], color='red',fill_color='white',alpha=0.8) # x_axis_type='datetime' für timeseries
output_file('circle.html')
show(p)

## Lines Plots
_ = p.line([1,2,3,4,5],[8,6,5,2,3],line_width=3)

## Patches
patch = np.array([[0,0],[10,0],[10,10],[0,10]]).transpose() # geht nicht
_ = p.patches(patch[0],patch[1]) # x,y sind lists von lists


#### Pandas
## ColumnDataSource
from bokeh.models import ColumnDataSource
#source = ColumnDataSource(data={'x':[1,2,3,4,5],'y':[8,6,5,2,3]})
source = ColumnDataSource(df) # kann dann als source für glyphs verwendet, dann
_ = p.circle(x='Year',y='Time',source=source,color='color',size=8)  # Spalten angeben
#_ = p.circle(x=df['hp'],y=df['mpg'],color=df['color']) # oder als parameter angeben
source.data


#### Tools

## Selection
p = figure(x_axis_label='x',y_axis_label='y',tools='box_select,lasso_select')
p.circle('petal_length','sepal_length',source=source,nonselection_fill_alpha=0.2,nonselection_fill_color='gray')
output_file('circle.html')
show(p)

## Hovering
from bokeh.models import HoverTool
hover = HoverTool(tooltips=None,mode='hline')
p = figure(tools=[hover,'crosshair'])
_ = p.circle([1,2,3,4,5],[8,6,5,2,3],hover_color='red')
output_file('circle.html')
show(p)

# Hover (mit Infos)
hover = HoverTool(tooltips=[
    ('Species Name', '@species'),
    ('Petal Length', '@petal_length')
])
p = figure(tools=[hover,'pan','wheel_zoom'])



#### Color Mapping
from bokeh.models import CategoricalColorMapper
mapper = CategoricalColorMapper(factors=['setosa','virginica','versicolor'],palette=['red','green','blue'])
p = figure()
source = ColumnDataSource(df)
_ = p.circle('petal_length','sepal_length',source=source,color={'field':'species','transform':mapper},legend='species')
output_file('circle.html')
show(p)


#### Layouts

## Row/Column-Layout
from bokeh.layouts import row,column
p1 = figure()
p2 = figure()
p3 = figure()
# einfach
layout = row(p1,p2)
# verschachtelt
row2 = column([p2,p3],sizing_mode='scale_width')
layout = row([p1,row2],sizing_mode='scale_width')
output_file('circle.html')
show(layout)

# Gridlayout
from bokeh.layouts import gridplot
layout = gridplot([[None,p1],[p2,p3]],toolbar_location=None)

# Tabs (Reiter)
from bokeh.models.widgets import Tabs,Panel
first = Panel(child=row(p1,p2), title='First Tab')
second = Panel(child=row(p3), title='Second Tab')
tabs = Tabs(tabs=[first,second])
output_file('tabbed.html')
show(tabs)

## Synchronisieren
# Axen - also Pan & Zoom - synchronisieren: Zooom & Pan im einen => auch im anderen
p3.x_range = p2.x_range = p1.x_range
p3.y_range = p2.y_range = p1.y_range
# Selektion sychnronisieren (Linked brushing): durch dieselbe Datenquelle teilen mit source=df

# Legende
# => Spaltennamen (wird automatisch erkannt) oder string für glyhen
p.circle('x', 'y', source=df, color=____, legend='string1')
p.circle('x', 'y', source=df2, color=____, legend='columnname')
p.legend.location = 'bottom_left'
p.legend.background_fill_color = 'lightgray'


#### Highlevel Charts

## Histograms
from bokeh.charts import Histogram

# Einfach
p = Histogram(df,'petal_length',title='Iris Morpholgy',bins=25)
# Überlagert nach Katergorien
p = Histogram(df,'petal_length',color='species',title='Iris Morpholgy',bins=25)
p.xaxis.axis_label = 'Petal Length'
p.yaxis.axis_label = 'Count (Female Length)'


## Boxplots
from bokeh.charts import BoxPlot
p = BoxPlot(df,values='petal_length',label='species',title='Iris Morphology')
# Shading
p = BoxPlot(df,values='petal_length',label='species',color='species',title='Iris Morphology')


## Scatter charts
# Erlaubt im gg zu bokeh.plotting direkt pandas und colors/marker-gruppierung
from bokeh.charts import Scatter
# Einfach
p = Scatter(df,x='petal_length',y='sepal_length',title='Irirs Morphology')
# Shading und Markers
p = Scatter(df,x='petal_length',y='sepal_length',color='species',marker='species',title='Irirs Morphology')
output_file('boxplot.html')
show(p)
