require 'Classifier'

function main()
    options = {}
    options.model = "CNN"
    clf = Classifier(options,100) 
    print (clf.net:__tostring())
end

main()
