"use client"

import type React from "react"
import { useState, useRef } from "react"
import {
  Upload,
  FileText,
  Brain,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  Play,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Separator } from "@/components/ui/separator"

export default function AiShayak() {
  const [uploadedModel, setUploadedModel] = useState<File | null>(null)
  const [uploadedDataset, setUploadedDataset] = useState<File | null>(null)
  const [sensitiveColumn, setSensitiveColumn] = useState<string>("")
  const [analysisComplete, setAnalysisComplete] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [analysisResults, setAnalysisResults] = useState<any>(null)
  const [errorMessage, setErrorMessage] = useState<string>("")
  const [evaluationType, setEvaluationType] = useState<"model" | "dataset">("model")
  const [availableColumns, setAvailableColumns] = useState<string[]>([])
  const [datasetPreview, setDatasetPreview] = useState<any>(null)

  const modelInputRef = useRef<HTMLInputElement>(null)
  const datasetInputRef = useRef<HTMLInputElement>(null)

  const handleModelUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && (file.name.endsWith(".pkl") || file.name.endsWith(".joblib"))) {
      setUploadedModel(file)
      setErrorMessage("")
    } else {
      alert("Please upload a valid model file (.pkl or .joblib)")
    }
  }

  const handleDatasetUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.name.endsWith(".csv")) {
      setUploadedDataset(file)
      previewDataset(file)
      setErrorMessage("")
    } else {
      alert("Please upload a valid CSV file")
    }
  }

  const previewDataset = async (file: File) => {
    try {
      const text = await file.text()
      const lines = text.split("\n")
      const headers = lines[0].split(",")
      const previewData = lines.slice(1, 6).map((line) => {
        const values = line.split(",")
        const row: any = {}
        headers.forEach((header, index) => {
          row[header.trim()] = values[index]?.trim() || ""
        })
        return row
      })
      setAvailableColumns(headers.map((h) => h.trim()))
      setDatasetPreview(previewData)
    } catch (error) {
      console.error("Error previewing dataset:", error)
    }
  }

  const startAnalysis = async () => {
    if (evaluationType === "model" && (!uploadedModel || !uploadedDataset || !sensitiveColumn.trim())) {
      setErrorMessage("Please upload both model and dataset files and specify sensitive column")
      return
    }
    if (evaluationType === "dataset" && (!uploadedDataset || !sensitiveColumn.trim())) {
      setErrorMessage("Please upload dataset file and specify sensitive column")
      return
    }

    setIsAnalyzing(true)
    setAnalysisProgress(0)
    setErrorMessage("")
    setAnalysisComplete(false)

    const formData = new FormData()
    if (uploadedModel) formData.append("modelFile", uploadedModel)
    if (uploadedDataset) formData.append("datasetFile", uploadedDataset)
    formData.append("sensitiveColumn", sensitiveColumn)
    formData.append("evaluationType", evaluationType)

    try {
      const steps = [20, 40, 60, 80, 100]
      for (const step of steps) {
        await new Promise((resolve) => setTimeout(resolve, 500))
        setAnalysisProgress(step)
      }

      const response = await fetch("http://localhost:5000/evaluate", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Analysis failed")
      }

      const results = await response.json()
      setAnalysisResults(results)
      setAnalysisComplete(true)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Analysis failed")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getDisplayResults = () => {
    return (
      analysisResults || {
        fairnessScore: 0,
        transparencyScore: 0,
        biasDetected: false,
        riskLevel: "Unknown",
        demographicParityDifference: 0,
        equalizedOddsDifference: 0,
        recommendations: [],
      }
    )
  }

  return (
    <div className="min-h-screen bg-background flex flex-col items-center text-white">
      <header className="border-b border-border bg-card w-full">
        <div className="container mx-auto px-8 py-6 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center justify-center w-12 h-12 bg-primary rounded-2xl">
              <Brain className="w-7 h-7 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">AI Sahayak</h1>
              <p className="text-sm text-gray-300">Cerebrators</p>
            </div>
          </div>
        </div>
      </header>

      <main className="w-full flex flex-col items-center py-12 space-y-12">
        <div className="text-center space-y-6 mb-12 px-8">
          <h2 className="text-5xl font-bold">Ethical AI Report Card</h2>
          <p className="text-xl text-gray-300 max-w-5xl mx-auto leading-relaxed">
            Upload your ML model and dataset to receive comprehensive bias detection, fairness analysis, and transparency scoring using scikit-learn and fairlearn libraries.
          </p>
        </div>

        <div className="w-full flex flex-col items-center space-y-10">
          <Card className="border-2 border-border bg-card/50 backdrop-blur-sm w-[95%] max-w-[1600px]">
            <CardHeader className="pb-6">
              <CardTitle className="flex items-center space-x-3 text-2xl">
                <Upload className="w-6 h-6" />
                <span>Upload Your AI Model & Dataset</span>
              </CardTitle>
              <CardDescription className="text-base text-gray-300">
                Upload your trained model (.pkl, .joblib) and test dataset (.csv) for real-time bias analysis
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-8">
              <div className="space-y-6">
                <div className="space-y-4">
                  <Label className="text-lg font-medium">Evaluation Type</Label>
                  <RadioGroup value={evaluationType} onValueChange={(value: "model" | "dataset") => setEvaluationType(value)}>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="model" id="model" />
                      <Label htmlFor="model" className="text-base">Model + Dataset Evaluation</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="dataset" id="dataset" />
                      <Label htmlFor="dataset" className="text-base">Dataset Analysis Only</Label>
                    </div>
                  </RadioGroup>
                </div>

                <Separator />

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <div className="border-2 border-dashed border-border rounded-2xl p-10 text-center hover:border-primary transition-all duration-300 hover:bg-card/30">
                    <Brain className="w-10 h-10 text-gray-400 mx-auto mb-4" />
                    <p className="text-lg font-medium mb-2">ML Model File</p>
                    <p className="text-sm text-gray-400 mb-4">Supports .pkl, .joblib formats</p>
                    <input ref={modelInputRef} type="file" accept=".pkl,.joblib" onChange={handleModelUpload} className="hidden" disabled={evaluationType === "dataset"} />
                    <Button variant="outline" size="lg" onClick={() => modelInputRef.current?.click()} disabled={evaluationType === "dataset"} className="w-full">
                      {uploadedModel ? uploadedModel.name : "Choose Model File"}
                    </Button>
                    {uploadedModel && (
                      <div className="mt-3 flex items-center justify-center">
                        <CheckCircle className="w-5 h-5 mr-2" />
                        <span className="text-sm font-medium">Model Loaded</span>
                      </div>
                    )}
                  </div>

                  <div className="border-2 border-dashed border-border rounded-2xl p-10 text-center hover:border-primary transition-all duration-300 hover:bg-card/30">
                    <FileText className="w-10 h-10 text-gray-400 mx-auto mb-4" />
                    <p className="text-lg font-medium mb-2">Test Dataset</p>
                    <p className="text-sm text-gray-400 mb-4">CSV format with features & labels</p>
                    <input ref={datasetInputRef} type="file" accept=".csv" onChange={handleDatasetUpload} className="hidden" />
                    <Button variant="outline" size="lg" onClick={() => datasetInputRef.current?.click()} className="w-full">
                      {uploadedDataset ? uploadedDataset.name : "Choose Dataset File"}
                    </Button>
                    {uploadedDataset && (
                      <div className="mt-3 flex items-center justify-center">
                        <CheckCircle className="w-5 h-5 mr-2" />
                        <span className="text-sm font-medium">Dataset Loaded</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {(uploadedModel || uploadedDataset) && (
                <div className="space-y-6 p-8 bg-muted/30 rounded-2xl">
                  <div className="space-y-4">
                    <Label htmlFor="sensitive-column" className="text-lg font-medium">Sensitive Feature Column</Label>
                    <Input id="sensitive-column" placeholder="e.g., gender, age_group, race" value={sensitiveColumn} onChange={(e) => setSensitiveColumn(e.target.value)} className="text-lg h-12" list="column-suggestions" />
                    <datalist id="column-suggestions">
                      {availableColumns.map((col) => (
                        <option key={col} value={col} />
                      ))}
                    </datalist>
                  </div>
                  <Button onClick={startAnalysis} size="lg" className="w-full h-12 text-lg font-medium" disabled={isAnalyzing}>
                    {isAnalyzing ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground mr-2"></div>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5 mr-2" />
                        Start Evaluation
                      </>
                    )}
                  </Button>
                </div>
              )}

              {errorMessage && (
                <Alert className="border border-white bg-muted/20 text-white">
                  <AlertTriangle className="h-5 w-5 text-white" />
                  <AlertTitle className="text-white">Error</AlertTitle>
                  <AlertDescription className="text-white">{errorMessage}</AlertDescription>
                </Alert>
              )}

              {isAnalyzing && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-base">
                    <span>Running analysis...</span>
                    <span className="font-medium">{analysisProgress}%</span>
                  </div>
                  <Progress value={analysisProgress} className="w-full h-3" />
                </div>
              )}
            </CardContent>
          </Card>

          {analysisComplete && (
            <>
              <Card className="border-2 border-border bg-card/50 backdrop-blur-sm w-[95%] max-w-[1600px] text-center">
                <CardHeader>
                  <CardTitle className="flex items-center justify-center space-x-3 text-2xl">
                    <BarChart3 className="w-6 h-6" />
                    <span>AI Ethics Report Card</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="overview" className="w-full">
                    <TabsList className="w-full h-12 flex justify-center">
                      <TabsTrigger value="overview" className="text-base">Overview</TabsTrigger>
                    </TabsList>
                    <TabsContent value="overview" className="space-y-8 mt-8">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full">
                        <Card className="bg-muted/20 w-full">
                          <CardContent className="p-10 text-center">
                            <div className="text-5xl font-bold mb-2">{getDisplayResults().fairnessScore}/10</div>
                            <p className="text-lg text-gray-300">Fairness Score</p>
                          </CardContent>
                        </Card>
                        <Card className="bg-muted/20 w-full">
                          <CardContent className="p-10 text-center">
                            <div className="text-5xl font-bold mb-2">{getDisplayResults().transparencyScore}/10</div>
                            <p className="text-lg text-gray-300">Transparency Score</p>
                          </CardContent>
                        </Card>
                        <Card className="bg-muted/20 w-full">
                          <CardContent className="p-10 text-center">
                            <Badge variant="secondary" className="text-white border-white text-lg px-4 py-2">
                              {getDisplayResults().riskLevel} Risk
                            </Badge>
                            <p className="text-lg text-gray-300 mt-3">Overall Risk</p>
                          </CardContent>
                        </Card>
                      </div>

                      {getDisplayResults().biasDetected && (
                        <Alert className="border border-white bg-muted/20 text-white">
                          <AlertTriangle className="h-5 w-5" />
                          <AlertTitle className="text-lg text-white">Bias Detected</AlertTitle>
                          <AlertDescription className="text-white">
                            Demographic Parity Difference: {getDisplayResults().demographicParityDifference} (threshold: 0.1). This model shows significant bias against certain demographic groups.
                          </AlertDescription>
                        </Alert>
                      )}
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>

              <Card className="bg-muted/20 border-2 border-border w-[95%] max-w-[1600px] text-center">
                <CardHeader>
                  <CardTitle className="text-2xl">Recommendations</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-4">
                    {getDisplayResults().recommendations.map((rec: string, index: number) => (
                      <li key={index} className="flex items-start justify-center space-x-3">
                        <div className="w-2 h-2 bg-primary rounded-full mt-3 flex-shrink-0" />
                        <span className="text-lg text-gray-300">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </main>
    </div>
  )
}
