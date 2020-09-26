do
  ws <- initWeights (length trainImages) (length trainLabels)
  let classes = classifyAll trainImages ws 
      --acc = accuracy classes trainLabels
  print classes 
