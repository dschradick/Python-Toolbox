########## VISUALISIERUNG - GGPLOT2
# Unterstüzung ist allerdings noch nicht so gut
from ggplot import *

mtcars.head(1)

#### Beachten:
# '' müssen benutzt werden
# \ vor zeilenende
# geom_smooth() gibts nicht

## Beisspiel Plot
ggplot(mtcars, aes(x='mpg', y='hp')) + geom_point()
ggplot(mtcars, aes(x='mpg', y='hp')) + \
    geom_point() + \
    stat_smooth(span=0.10, color='blue', se=True) + \
    ggtitle('ggplot is great!')
