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
  ShieldCheck,
  Search,
  Users,
  Download,
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
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { Bar, BarChart, XAxis, YAxis, ResponsiveContainer, Cell } from "recharts"

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
        await new Promise((resolve) => setTimeout(resolve, 300))
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

  const downloadReport = () => {
    if (!analysisResults) return;

    const res = analysisResults;
    let report = `AI SHAYAK - ETHICAL AI AUDIT REPORT\n`;
    report += `====================================\n\n`;
    report += `OVERALL STATUS\n`;
    report += `--------------\n`;
    report += `Ethics Score: ${res.ethicsScore}/10\n`;
    report += `Risk Level: ${res.riskLevel}\n`;
    report += `Fairness Score: ${res.fairnessScore}/10\n`;
    report += `Data Integrity Score: ${res.integrityScore}/10\n`;
    report += `Transparency Score: ${res.transparencyScore}/10\n\n`;

    report += `METRICS\n`;
    report += `-------\n`;
    report += `Demographic Parity Difference: ${res.metrics.demographicParityDifference}\n`;
    report += `Equalized Odds Difference: ${res.metrics.equalizedOddsDifference}\n`;
    report += `Disparate Impact Ratio: ${res.metrics.disparateImpact}\n`;
    report += `Robustness: ${(res.metrics.robustness * 100).toFixed(1)}%\n`;
    report += `Privacy: ${(res.metrics.privacy * 100).toFixed(1)}%\n\n`;

    report += `COMPLIANCE\n`;
    report += `----------\n`;
    res.compliance?.frameworks?.forEach((fw: any) => {
      report += `${fw.name}: ${fw.status}\n`;
      report += `Requirement: ${fw.requirement}\n`;
      report += `Details: ${fw.details}\n\n`;
    });

    report += `MODEL CARD\n`;
    report += `----------\n`;
    report += `Intended Use: ${res.compliance?.modelCard?.intendedUse}\n`;
    report += `Fairness Philosophy: ${res.compliance?.modelCard?.fairnessPhilosophy}\n`;
    report += `Limitations: ${res.compliance?.modelCard?.limitations}\n\n`;

    report += `RECOMMENDATIONS\n`;
    report += `---------------\n`;
    res.recommendations.forEach((rec: string, i: number) => {
      report += `${i + 1}. ${rec}\n`;
    });
    report += `\n`;

    report += `GROUP PERFORMANCE\n`;
    report += `-----------------\n`;
    Object.entries(res.groupPerformance).forEach(([group, data]: [string, any]) => {
      report += `Group: ${group}\n`;
      if (data.selectionRate !== undefined) {
        report += ` - Selection Rate: ${data.selectionRate.toFixed(3)}\n`;
        report += ` - Accuracy: ${(data.accuracy * 100).toFixed(1)}%\n`;
      } else {
        report += ` - Count: ${data.count}\n`;
        report += ` - Proportion: ${(data.percentage * 100).toFixed(1)}%\n`;
      }
      report += `\n`;
    });

    const blob = new Blob([report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `AI_Shayak_Audit_Report_${new Date().toISOString().split("T")[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const getDisplayResults = () => {
    return (
      analysisResults || {
        ethicsScore: 0,
        fairnessScore: 0,
        integrityScore: 0,
        transparencyScore: 0,
        biasDetected: false,
        riskLevel: "Unknown",
        metrics: {
          demographicParityDifference: 0,
          equalizedOddsDifference: 0,
          disparateImpact: 0,
          statisticalParity: 0,
          robustness: 0,
          privacy: 0
        },
        compliance: {
          frameworks: [],
          modelCard: { intendedUse: "", limitations: "", fairnessPhilosophy: "" }
        },
        recommendations: [],
        chartData: [],
        groupPerformance: {}
      }
    )
  }

  const chartConfig = {
    value: {
      label: evaluationType === "model" ? "Selection Rate" : "Representation %",
      color: "hsl(var(--primary))",
    },
  }

  return (
    <div className="min-h-screen bg-background flex flex-col items-center text-white">
      <header className="border-b border-border bg-card w-full">
        <div className="container mx-auto px-8 py-6 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center justify-center w-12 h-12 bg-primary rounded-2xl">
              <ShieldCheck className="w-7 h-7 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">AI Shayak</h1>
              <p className="text-sm text-gray-300">Ethical AI Platform</p>
            </div>
          </div>
        </div>
      </header>

      <main className="w-full flex flex-col items-center py-12 space-y-12">
        <div className="text-center space-y-6 mb-12 px-8">
          <h2 className="text-5xl font-bold">AI Ethics & Bias Audit</h2>
          <p className="text-xl text-gray-300 max-w-5xl mx-auto leading-relaxed">
            Our platform evaluates models and datasets directly to provide deep ethics scoring, fairness auditing, and transparency reporting without relying on external evaluation models.
          </p>
        </div>

        <div className="w-full flex flex-col items-center space-y-10">
          <Card className="border-2 border-border bg-card/50 backdrop-blur-sm w-[95%] max-w-[1600px]">
            <CardHeader className="pb-6">
              <CardTitle className="flex items-center space-x-3 text-2xl">
                <Upload className="w-6 h-6" />
                <span>Input Assets for Audit</span>
              </CardTitle>
              <CardDescription className="text-base text-gray-300">
                Upload your local machine learning artifacts for immediate ethical evaluation.
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-8">
              <div className="space-y-6">
                <div className="space-y-4">
                  <Label className="text-lg font-medium">Audit Scope</Label>
                  <RadioGroup value={evaluationType} onValueChange={(value: "model" | "dataset") => setEvaluationType(value)}>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="model" id="model" />
                      <Label htmlFor="model" className="text-base">End-to-End Audit (Model + Dataset)</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="dataset" id="dataset" />
                      <Label htmlFor="dataset" className="text-base">Data-Only Ethics Audit</Label>
                    </div>
                  </RadioGroup>
                </div>

                <Separator />

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <div className="border-2 border-dashed border-border rounded-2xl p-10 text-center hover:border-primary transition-all duration-300 hover:bg-card/30">
                    <Brain className="w-10 h-10 text-gray-400 mx-auto mb-4" />
                    <p className="text-lg font-medium mb-2">Model Artifact</p>
                    <p className="text-sm text-gray-400 mb-4">Supports .pkl, .joblib (Scikit-learn pipelines)</p>
                    <input ref={modelInputRef} type="file" accept=".pkl,.joblib" onChange={handleModelUpload} className="hidden" disabled={evaluationType === "dataset"} />
                    <Button variant="outline" size="lg" onClick={() => modelInputRef.current?.click()} disabled={evaluationType === "dataset"} className="w-full">
                      {uploadedModel ? uploadedModel.name : "Select Model"}
                    </Button>
                  </div>

                  <div className="border-2 border-dashed border-border rounded-2xl p-10 text-center hover:border-primary transition-all duration-300 hover:bg-card/30">
                    <FileText className="w-10 h-10 text-gray-400 mx-auto mb-4" />
                    <p className="text-lg font-medium mb-2">Dataset Samples</p>
                    <p className="text-sm text-gray-400 mb-4">CSV format for fairness testing</p>
                    <input ref={datasetInputRef} type="file" accept=".csv" onChange={handleDatasetUpload} className="hidden" />
                    <Button variant="outline" size="lg" onClick={() => datasetInputRef.current?.click()} className="w-full">
                      {uploadedDataset ? uploadedDataset.name : "Select Dataset"}
                    </Button>
                  </div>
                </div>
              </div>

              {(uploadedModel || uploadedDataset) && (
                <div className="space-y-6 p-8 bg-muted/30 rounded-2xl">
                  <div className="space-y-4">
                    <Label htmlFor="sensitive-column" className="text-lg font-medium">Define Sensitive Attribute</Label>
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <Input id="sensitive-column" placeholder="e.g., race, gender, age" value={sensitiveColumn} onChange={(e) => setSensitiveColumn(e.target.value)} className="pl-10 text-lg h-12" list="column-suggestions" />
                    </div>
                    <datalist id="column-suggestions">
                      {availableColumns.map((col) => (
                        <option key={col} value={col} />
                      ))}
                    </datalist>
                    <p className="text-sm text-gray-400">Specify the column name that represents the demographic group you want to audit for fairness.</p>
                  </div>
                  <Button onClick={startAnalysis} size="lg" className="w-full h-12 text-lg font-medium" disabled={isAnalyzing}>
                    {isAnalyzing ? "Processing Audit..." : "Run Ethics Evaluation"}
                  </Button>
                </div>
              )}

              {errorMessage && (
                <Alert variant="destructive">
                  <AlertTriangle className="h-5 w-5" />
                  <AlertTitle>Audit Failed</AlertTitle>
                  <AlertDescription>{errorMessage}</AlertDescription>
                </Alert>
              )}

              {isAnalyzing && (
                <div className="space-y-3">
                  <Progress value={analysisProgress} className="w-full h-3" />
                </div>
              )}
            </CardContent>
          </Card>

          {analysisComplete && (
            <div className="w-[95%] max-w-[1600px] space-y-8">
              <div className="flex justify-between items-center">
                <h3 className="text-3xl font-bold">Audit Results</h3>
                <Button onClick={downloadReport} variant="outline" className="flex items-center space-x-2">
                  <Download className="w-5 h-5" />
                  <span>Download Full Audit Report</span>
                </Button>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                 <Card className="bg-card/50 border-2 border-primary">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg text-gray-300">Overall Ethics Score</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-5xl font-bold">{getDisplayResults().ethicsScore}/10</div>
                      <Badge className="mt-2" variant={getDisplayResults().riskLevel === "Low" ? "default" : getDisplayResults().riskLevel === "Medium" ? "outline" : "destructive"}>
                        {getDisplayResults().riskLevel} Risk
                      </Badge>
                    </CardContent>
                 </Card>
                 <Card className="bg-card/30 border-2 border-border">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg text-gray-300">Fairness</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-4xl font-bold">{getDisplayResults().fairnessScore}/10</div>
                      <Progress value={getDisplayResults().fairnessScore * 10} className="mt-3 h-2" />
                    </CardContent>
                 </Card>
                 <Card className="bg-card/30 border-2 border-border">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg text-gray-300">Data Integrity</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-4xl font-bold">{getDisplayResults().integrityScore}/10</div>
                      <Progress value={getDisplayResults().integrityScore * 10} className="mt-3 h-2" />
                    </CardContent>
                 </Card>
                 <Card className="bg-card/30 border-2 border-border">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg text-gray-300">Transparency</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-4xl font-bold">{getDisplayResults().transparencyScore}/10</div>
                      <Progress value={getDisplayResults().transparencyScore * 10} className="mt-3 h-2" />
                    </CardContent>
                 </Card>
              </div>

              <Card className="border-2 border-border bg-card/50 backdrop-blur-sm">
                <CardContent className="p-8">
                  <Tabs defaultValue="bias" className="w-full">
                    <TabsList className="grid w-full grid-cols-4 h-14">
                      <TabsTrigger value="bias" className="text-lg"><AlertTriangle className="w-5 h-5 mr-2"/> Bias & Fairness</TabsTrigger>
                      <TabsTrigger value="security" className="text-lg"><ShieldCheck className="w-5 h-5 mr-2"/> Security Audit</TabsTrigger>
                      <TabsTrigger value="groups" className="text-lg"><Users className="w-5 h-5 mr-2"/> Group Analysis</TabsTrigger>
                      <TabsTrigger value="compliance" className="text-lg"><FileText className="w-5 h-5 mr-2"/> Compliance</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="bias" className="pt-8 space-y-8">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-6">
                           <div className="p-6 bg-muted/20 rounded-2xl border border-border">
                              <h4 className="text-xl font-semibold mb-4">Parity Metrics</h4>
                              <div className="space-y-4">
                                 <div className="flex justify-between items-center">
                                    <span className="text-gray-300">Demographic Parity Diff</span>
                                    <span className="font-mono text-xl">{getDisplayResults().metrics.demographicParityDifference}</span>
                                 </div>
                                 <Separator className="bg-border/50" />
                                 <div className="flex justify-between items-center">
                                    <span className="text-gray-300">Equalized Odds Diff</span>
                                    <span className="font-mono text-xl">{getDisplayResults().metrics.equalizedOddsDifference}</span>
                                 </div>
                              </div>
                           </div>
                        </div>
                        <div className="space-y-6">
                           <div className="p-6 bg-muted/20 rounded-2xl border border-border">
                              <h4 className="text-xl font-semibold mb-4">Impact Ratios</h4>
                              <div className="space-y-4">
                                 <div className="flex justify-between items-center">
                                    <span className="text-gray-300">Disparate Impact Ratio</span>
                                    <span className={`font-mono text-xl ${getDisplayResults().metrics.disparateImpact < 0.8 ? 'text-red-400' : 'text-green-400'}`}>
                                      {getDisplayResults().metrics.disparateImpact}
                                    </span>
                                 </div>
                                 <Separator className="bg-border/50" />
                                 <div className="flex justify-between items-center">
                                    <span className="text-gray-300">Mitigation Strategy</span>
                                    <Badge variant="outline" className="text-primary border-primary">Post-Processing</Badge>
                                 </div>
                              </div>
                           </div>
                        </div>
                      </div>

                      <div className="p-6 bg-primary/10 rounded-2xl border border-primary/30">
                        <h4 className="text-xl font-semibold mb-3 flex items-center">
                          <Brain className="w-6 h-6 mr-2 text-primary" />
                          Automated Mitigation Plan
                        </h4>
                        <p className="text-gray-300 text-lg">
                          {getDisplayResults().recommendations[0] || "No critical bias mitigation required based on current parity levels."}
                        </p>
                      </div>
                    </TabsContent>

                    <TabsContent value="security" className="pt-8 space-y-8">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <Card className="bg-card/30 border-border">
                          <CardHeader>
                            <CardTitle className="flex items-center">
                              <ShieldCheck className="w-5 h-5 mr-2 text-blue-400" />
                              Robustness Audit
                            </CardTitle>
                            <CardDescription>Sensitivity to input perturbations</CardDescription>
                          </CardHeader>
                          <CardContent>
                            <div className="text-4xl font-bold mb-2">{(getDisplayResults().metrics.robustness * 100).toFixed(1)}%</div>
                            <p className="text-sm text-gray-400">Model stability when numerical inputs are perturbed by ±5% Gaussian noise.</p>
                            <Progress value={getDisplayResults().metrics.robustness * 100} className="mt-4 h-2" />
                          </CardContent>
                        </Card>
                        <Card className="bg-card/30 border-border">
                          <CardHeader>
                            <CardTitle className="flex items-center">
                              <Search className="w-5 h-5 mr-2 text-purple-400" />
                              Privacy Leakage
                            </CardTitle>
                            <CardDescription>Attribute inference risk</CardDescription>
                          </CardHeader>
                          <CardContent>
                            <div className="text-4xl font-bold mb-2">{(getDisplayResults().metrics.privacy * 100).toFixed(1)}%</div>
                            <p className="text-sm text-gray-400">Privacy Score: Measures how difficult it is to guess the sensitive attribute from model outcomes.</p>
                            <Progress value={getDisplayResults().metrics.privacy * 100} className="mt-4 h-2" />
                          </CardContent>
                        </Card>
                      </div>
                    </TabsContent>

                    <TabsContent value="compliance" className="pt-8 space-y-6">
                       <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                          {getDisplayResults().compliance?.frameworks?.map((fw: any, idx: number) => (
                            <div key={idx} className="p-6 bg-muted/20 rounded-2xl border border-border">
                               <div className="flex justify-between items-start mb-4">
                                  <h4 className="text-xl font-bold">{fw.name}</h4>
                                  <Badge variant={fw.status === 'Compliant' ? 'default' : 'destructive'}>{fw.status}</Badge>
                               </div>
                               <p className="text-primary text-sm font-semibold mb-2">{fw.requirement}</p>
                               <p className="text-gray-400">{fw.details}</p>
                            </div>
                          ))}
                       </div>
                       
                       <Card className="bg-card/30 border-dashed border-2 border-border">
                          <CardHeader>
                             <CardTitle>Generated Model Card (Ethics Metadata)</CardTitle>
                          </CardHeader>
                          <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-6">
                             <div>
                                <h5 className="text-sm font-bold text-gray-500 uppercase mb-2">Intended Use</h5>
                                <p className="text-gray-300">{getDisplayResults().compliance?.modelCard?.intendedUse}</p>
                             </div>
                             <div>
                                <h5 className="text-sm font-bold text-gray-500 uppercase mb-2">Fairness Philosophy</h5>
                                <p className="text-gray-300">{getDisplayResults().compliance?.modelCard?.fairnessPhilosophy}</p>
                             </div>
                             <div>
                                <h5 className="text-sm font-bold text-gray-500 uppercase mb-2">Limitations</h5>
                                <p className="text-gray-300">{getDisplayResults().compliance?.modelCard?.limitations}</p>
                             </div>
                          </CardContent>
                       </Card>
                    </TabsContent>

                    <TabsContent value="groups" className="pt-8">
                       <div className="grid grid-cols-1 xl:grid-cols-2 gap-10">
                          <div className="h-[400px]">
                             <h4 className="text-xl font-semibold mb-6 text-center">
                               {evaluationType === "model" ? "Selection Rate by Group" : "Dataset Distribution"}
                             </h4>
                             <ChartContainer config={chartConfig} className="h-full w-full">
                                <BarChart data={getDisplayResults().chartData}>
                                   <XAxis dataKey="group" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                                   <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}${evaluationType === "model" ? "" : "%"}`} />
                                   <ChartTooltip content={<ChartTooltipContent />} />
                                   <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                                      {getDisplayResults().chartData.map((entry: any, index: number) => (
                                        <Cell key={`cell-${index}`} fill={index % 2 === 0 ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.6)"} />
                                      ))}
                                   </Bar>
                                </BarChart>
                             </ChartContainer>
                          </div>
                          <div className="overflow-hidden rounded-2xl border border-border">
                             <table className="w-full text-left">
                                <thead className="bg-muted/50">
                                   <tr>
                                      <th className="p-4 font-semibold">Group</th>
                                      <th className="p-4 font-semibold">{evaluationType === "model" ? "Rate" : "Count"}</th>
                                      <th className="p-4 font-semibold">{evaluationType === "model" ? "Accuracy" : "Prop."}</th>
                                   </tr>
                                </thead>
                                <tbody>
                                   {Object.entries(getDisplayResults().groupPerformance).map(([group, data]: [string, any]) => (
                                     <tr key={group} className="border-t border-border hover:bg-muted/20">
                                        <td className="p-4">{group}</td>
                                        <td className="p-4 font-mono">{evaluationType === "model" ? data.selectionRate.toFixed(3) : data.count}</td>
                                        <td className="p-4 font-mono">
                                          {evaluationType === "model" ? (data.accuracy * 100).toFixed(1) + "%" : (data.percentage * 100).toFixed(1) + "%"}
                                        </td>
                                     </tr>
                                   ))}
                                </tbody>
                             </table>
                          </div>
                       </div>
                    </TabsContent>

                    <TabsContent value="recommendations" className="pt-8">
                       <div className="space-y-4">
                          {getDisplayResults().recommendations.map((rec: string, index: number) => (
                            <div key={index} className="flex items-start p-4 bg-muted/20 rounded-xl border-l-4 border-primary">
                               <CheckCircle className="w-6 h-6 mr-4 text-primary shrink-0" />
                               <span className="text-lg text-gray-200">{rec}</span>
                            </div>
                          ))}
                          {getDisplayResults().recommendations.length === 0 && (
                            <div className="text-center py-10 text-gray-400">
                               No critical recommendations. The model appears to meet baseline ethical standards.
                            </div>
                          )}
                       </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </main>

      <footer className="w-full border-t border-border py-8 mt-20 bg-card/30">
        <div className="container mx-auto px-8 text-center text-gray-400">
           <p>© 2026 AI Shayak - Empowering Ethical Machine Learning</p>
        </div>
      </footer>
    </div>
  )
}
