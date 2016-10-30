function evaluate(testData)
	assert(paths.filep("pcanet.t7") and paths.filep("model.t7"),"pcanet or model does not exist")
	pcanet = torch.load("pcanet.t7")
	model = torch.load("model.t7")

    local test_accuracy = 0
    model:evaluate() -- turn on the evaluation mode

    -- Perform the forward pass (prediction mode).
    local predictions = model:forward(pcanet:PCANet_FeaExt(testData.data))
    -- evaluate results.
    for i = 1, predictions:size(1) do
        local _, predicted_label = predictions[i]:max(1)
        if predicted_label[1] == testData.labels[i] then test_accuracy = test_accuracy + 1 end
    end

    test_accuracy = test_accuracy / (testData.data:size(1))
    return test_accuracy
end