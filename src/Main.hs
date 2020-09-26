import Data.Word
import Data.Ord
import Control.Monad
import Control.Applicative
import System.Random

import qualified Data.Vector.Unboxed as U
import qualified Data.Vector as V
import qualified Data.ByteString as BS

-- Training images
trainImageFile = "./data/train-images-idx3-ubyte"

-- Training labels
trainLabelFile = "./data/train-labels-idx1-ubyte"

-- Test images
testImageFile = "./data/t10k-images-idx3-ubyte"

-- Test labels
testLabelFile = "./data/t10k-labels-idx1-ubyte"


-- Type synonyms
-- TODO: Replace with newtype declarations

type Label = Word8

-- These can be unboxed vectors (hold raw data values)
type LabelVector = U.Vector Label
type ImageVector = U.Vector Double
type WeightVector = U.Vector Double

-- These need to be boxed vectors (holds pointers to unboxed vectors)
type ImageMatrix = V.Vector ImageVector

main :: IO ()
main = do
  -- Read bytestrings directly from files
  putStrLn "Parsing training data..."
  trainImageData <- BS.readFile trainImageFile
  trainLabelData <- BS.readFile trainLabelFile
  putStrLn "Parsing test data..."
  testImageData <- BS.readFile testImageFile
  testLabelData <- BS.readFile testLabelFile

  let trainImages = getData 60000 trainImageData
      testImages = getData 10000 testImageData

      trainLabels = getLabels 60000 trainLabelData
      testLabels = getLabels 10000 testLabelData

  weights <- initWeights 

  putStr "Training images: "
  print $ V.length trainImages
  putStr "Test images: "
  print $ V.length testImages

  print $ U.length weights

getData :: Int -> BS.ByteString -> ImageMatrix
getData i bs = 
  -- first 16 bytes are header
  let bytes = BS.drop 16 bs
  in V.generate i $ \n ->
    U.generate 785 $ \m ->
      if m == 0 then 1.0 
      else (/255) . fromIntegral $ 
        BS.index bytes $ n*784 + (m-1)

getLabels :: Int -> BS.ByteString -> LabelVector
getLabels i bs =
  -- first 8 bytes are header
  let bytes = BS.drop 8 bs
  in U.generate i $ \n -> 
    fromIntegral $ BS.index bytes n

initWeights :: IO WeightVector
initWeights = do
  g <- getStdGen
  let rs = randomRs (-0.05, 0.05) g :: [Double]
  pure $ U.fromList $ take 785 rs

{--
 
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


chunksOf :: Int -> [a] -> [[a]]
chunksOf n xs =
  case xs of
    [] -> []
    _ -> (take n xs) : (chunksOf n $ drop n xs)

--}
