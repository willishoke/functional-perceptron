import Data.Word
import Data.List
import Data.Ord
import Control.Monad
import Control.Applicative
import System.Random

import qualified Data.ByteString as BS

trainImageFile = "./data/train-images-idx3-ubyte"
trainLabelFile = "./data/train-labels-idx1-ubyte"
testImageFile = "./data/t10k-images-idx3-ubyte"
testLabelFile = "./data/t10k-labels-idx1-ubyte"

type Label = Word8
type Labels = [Label]
type Image = [Double]
type Images = [Image]
type Weight = [Double]
type Weights = [Weight]

main :: IO ()
main = do
  putStrLn "Parsing training data..."
  trainImageData <- BS.readFile trainImageFile
  trainLabelData <- BS.readFile trainLabelFile
  putStrLn "Parsing test data..."
  testImageData <- BS.readFile testImageFile
  testLabelData <- BS.readFile testLabelFile
  let trainImages = getData trainImageData
      trainLabels = getLabels trainLabelData
      testImages = getData testImageData
      testLabels = getLabels testLabelData
  putStr "Training images: "
  print $ length trainImages
  putStr "Test images: "
  print $ length testImages
  ws <- initWeights (length trainImages) (length trainLabels)
  let classes = classifyAll trainImages ws 
      --acc = accuracy classes trainLabels
  print classes 
  print "done" 


dotProduct :: Image -> Weight -> Double
dotProduct i w = sum $ zipWith (*) i w

classify :: Image -> Weights -> Label
classify i ws =
  let dots = map (dotProduct i) ws
  in fromIntegral $ head $ elemIndices (maximum dots) dots

classifyAll :: Images -> Weights -> Labels
classifyAll imageData weights =
  map (\i -> classify i weights) imageData
    
accuracy :: Labels -> Labels -> Double
accuracy estLabels trueLabels =
  let diffs = filter id $ zipWith (==) estLabels trueLabels 
      correct = fromIntegral (length trueLabels)
      total = fromIntegral (length diffs)
  in correct / total
 
-- η

confusionMatrix :: Labels -> Labels -> [[Int]]
confusionMatrix actual predicted =
  let pairs = zip actual predicted
      labelEnum = nub $ sort actual
      possible = (liftA2 (,)) labelEnum labelEnum
      counts = \p -> length $ elemIndices p pairs
  in chunksOf (length labelEnum) $ map counts possible

chop :: Int -> String -> String
chop n =
  let pad n str = (++) (replicate (n - (length str)) ' ') str
  in pad n . take n

printMatrix :: [[Int]] -> IO ()
printMatrix m = do
  let transform = map (map $ chop 5 . show) m
  mapM_ putStrLn $ map concat transform

initWeights :: Int -> Int -> IO Weights
initWeights numWeights numClassifications = do
  g <- getStdGen
  let rs = randomRs (-0.5, 0.5) g :: [Double]
  pure $ take numClassifications $ chunksOf numWeights rs

chunksOf :: Int -> [a] -> [[a]]
chunksOf n xs =
  case xs of
    [] -> []
    _ -> (take n xs) : (chunksOf n $ drop n xs)

getData :: BS.ByteString -> Images
getData bs = 
  let imageData = BS.unpack $ BS.drop 16 bs
      f = (/255) . fromIntegral
  in map (1.0:) $ chunksOf (28*28) $ map f imageData 

getLabels :: BS.ByteString -> Labels
getLabels = BS.unpack . BS.drop 8
