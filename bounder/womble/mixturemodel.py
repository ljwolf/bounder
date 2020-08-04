import numpy

if __name__ == '__main__':
    import numpy, geopandas, pysal, pystan
    
    data = geopandas.read_file(pysal.lib.examples.get_path('columbus.shp'))
    
    connectivity = pysal.lib.weights.Rook.from_dataframe(data)
    X = data[['HOVAL', 'CRIME', 'INC']].values
    X[:,0] = numpy.log(X[:,0])
    
    n,p = X.shape
    
    max_K = 20
    
    known = numpy.concatenate((numpy.random.normal(-5,2,size=14), 
                               numpy.random.normal(2,1,size=49-14))).reshape(-1,1) 
    known_class = [1]*14 + [2]*(49-14)
