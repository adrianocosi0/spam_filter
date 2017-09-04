import datashader as ds

def plot_datashader(df,x,y,aggr=ds.reductions.count(),spread=False,
                    x_range=None,
                    y_range=None,
                    plot_width=800, plot_height=800//1.5,
                    back_color=None,cmap=['lightblue', 'darkblue'],
                   threshold=0.5):
    if x_range =='limit':
        x_range = (df[x].describe(percentiles=[0.05]).loc['5%'],df[x].describe(percentiles=[0.95]).loc['95%'])
    if y_range =='limit':
        y_range = (df[y].describe(percentiles=[0.05]).loc['5%'],df[y].describe(percentiles=[0.95]).loc['95%'])
    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                      x_range=x_range,
                      y_range=y_range)
    tf = canvas.points(df,x,y,agg=aggr)
    tf = ds.transfer_functions.shade(tf,cmap=cmap)
    if spread:
        tf = ds.transfer_functions.dynspread(tf,threshold=threshold)
    tf=ds.transfer_functions.set_background(tf,color=back_color)
    return tf
