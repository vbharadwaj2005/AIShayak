"use client"

import type React from "react"
import { useState, useRef, useCallback } from "react"
import {
  Upload, FileText, Brain, AlertTriangle, CheckCircle,
  ShieldCheck, Search, Users, Download, Sliders, Lock,
  Layers, Table2, Gauge, Target, ArrowUpDown, RefreshCw, BarChart3,
} from "lucide-react"
import { Button } from "@/components/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/card"
import { Progress } from "@/components/progress"
import { Badge } from "@/components/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/alert"
import { Input } from "@/components/input"
import { Label } from "@/components/label"
import { RadioGroup, RadioGroupItem } from "@/components/radio-group"
import { Separator } from "@/components/separator"
import {
  ChartContainer, ChartTooltip, ChartTooltipContent,
} from "@/components/chart"
import { Bar, BarChart, XAxis, YAxis, Cell } from "recharts"

const API_BASE = "http://localhost:5000"

export default function AiShayak() {
  const [uploadedModel, setUploadedModel] = useState<File | null>(null)
  const [uploadedDataset, setUploadedDataset] = useState<File | null>(null)
  const [sensitiveColumn, setSensitiveColumn] = useState<string>("")
  const [sensitiveColumnsMulti, setSensitiveColumnsMulti] = useState<string>("")
  const [analysisComplete, setAnalysisComplete] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [analysisResults, setAnalysisResults] = useState<any>(null)
  const [errorMessage, setErrorMessage] = useState<string>("")
  const [evaluationType, setEvaluationType] = useState<"model" | "dataset">("model")
  const [availableColumns, setAvailableColumns] = useState<string[]>([])
  const [schemaPreview, setSchemaPreview] = useState<any>(null)
  const [showSchema, setShowSchema] = useState(false)
  const [dragOver, setDragOver] = useState<string | null>(null)
  const [schemaLoading, setSchemaLoading] = useState(false)
  const [advancedResults, setAdvancedResults] = useState<Record<string, any>>({})
  const [isAdvancedLoading, setIsAdvancedLoading] = useState<Record<string, boolean>>({})
  const [activeAdvancedTab, setActiveAdvancedTab] = useState<string>("optimization")
  const [deepAuditType, setDeepAuditType] = useState<"intersectional" | "calibration">("intersectional")

  const modelInputRef = useRef<HTMLInputElement>(null)
  const datasetInputRef = useRef<HTMLInputElement>(null)

  const MODEL_EXTS = ".pkl,.joblib,.json,.ubj,.pt,.pth,.h5,.keras"

  const isValidModel = (name: string) =>
    /\.(pkl|joblib|json|ubj|pt|pth|h5|keras)$/i.test(name)

  const handleModelUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && isValidModel(file.name)) {
      setUploadedModel(file)
      setErrorMessage("")
    } else {
      alert("Please upload a valid model file (.pkl, .joblib, .json, .ubj, .pt, .pth, .h5, .keras)")
    }
  }

  const handleDatasetUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.name.endsWith(".csv")) {
      if (file.size > 500 * 1024 * 1024) {
        alert("Dataset file exceeds maximum size (500MB)")
        return
      }
      setUploadedDataset(file)
      readDatasetPreview(file)
      setErrorMessage("")
    } else {
      alert("Please upload a valid CSV file")
    }
  }

  const handleDrop = useCallback((type: "model" | "dataset", e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(null)
    const file = e.dataTransfer.files[0]
    if (!file) return
    if (type === "model") {
      if (!isValidModel(file.name)) { alert("Please drop a valid model file"); return }
      setUploadedModel(file)
    } else {
      if (!file.name.endsWith(".csv")) { alert("Please drop a valid CSV file"); return }
      if (file.size > 500 * 1024 * 1024) { alert("File exceeds 500MB"); return }
      setUploadedDataset(file)
      readDatasetPreview(file)
    }
    setErrorMessage("")
  }, [])

  const readDatasetPreview = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const text = e.target?.result as string
      const headers = text.split("\n")[0].split(",").map((h: string) => h.trim())
      setAvailableColumns(headers)
    }
    reader.readAsText(file.slice(0, 10240))
  }

  const loadSchemaPreview = async () => {
    if (!uploadedDataset) return
    setSchemaLoading(true)
    setShowSchema(true)
    const fd = new FormData()
    fd.append("datasetFile", uploadedDataset)
    try {
      const res = await fetch(`${API_BASE}/preview`, { method: "POST", body: fd })
      if (res.ok) setSchemaPreview(await res.json())
    } catch { /* non-critical */ }
    setSchemaLoading(false)
  }

  const startAnalysis = async () => {
    if (evaluationType === "model" && (!uploadedModel || !uploadedDataset || !sensitiveColumn.trim())) {
      setErrorMessage("Upload model, dataset, and specify sensitive column"); return
    }
    if (evaluationType === "dataset" && (!uploadedDataset || !sensitiveColumn.trim())) {
      setErrorMessage("Upload dataset and specify sensitive column"); return
    }
    setIsAnalyzing(true)
    setAnalysisProgress(10)
    setErrorMessage("")
    setAnalysisComplete(false)
    setAdvancedResults({})

    const fd = new FormData()
    if (uploadedModel) fd.append("modelFile", uploadedModel)
    if (uploadedDataset) fd.append("datasetFile", uploadedDataset)
    fd.append("sensitiveColumn", sensitiveColumn)
    fd.append("evaluationType", evaluationType)

    try {
      setAnalysisProgress(30)
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 120_000)
      const res = await fetch(`${API_BASE}/evaluate`, { method: "POST", body: fd, signal: controller.signal })
      clearTimeout(timeoutId)
      setAnalysisProgress(70)
      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.error || "Analysis failed")
      }
      setAnalysisProgress(100)
      setAnalysisResults(await res.json())
      setAnalysisComplete(true)
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setErrorMessage("Request timed out after 120 seconds")
      } else {
        setErrorMessage(error instanceof Error ? error.message : "Analysis failed")
      }
    } finally {
      setIsAnalyzing(false)
    }
  }

  const runAdvancedAudit = async (type: string) => {
    if (!uploadedModel || !uploadedDataset || !sensitiveColumn.trim()) {
      setErrorMessage("Upload model, dataset, and specify sensitive column first"); return
    }
    setIsAdvancedLoading((prev) => ({ ...prev, [type]: true }))
    setErrorMessage("")
    const fd = new FormData()
    fd.append("modelFile", uploadedModel)
    fd.append("datasetFile", uploadedDataset)

    let url = ""
    let extra: Record<string, string> = {}
    if (["intersectional", "calibration"].includes(type)) {
      url = `${API_BASE}/deep-audit`
      extra = { sensitiveColumns: type === "intersectional" ? `${sensitiveColumn},${sensitiveColumnsMulti || sensitiveColumn}` : sensitiveColumn, auditType: type }
    } else if (["threshold", "retraining", "regression-fairness"].includes(type)) {
      url = `${API_BASE}/optimize`
      extra = { sensitiveColumn, optimizeType: type }
    } else if (["adversarial", "differential-privacy", "membership-inference"].includes(type)) {
      url = `${API_BASE}/security-audit`
      extra = { sensitiveColumn, auditType: type }
    }
    Object.entries(extra).forEach(([k, v]) => fd.append(k, v))

    try {
      const res = await fetch(url, { method: "POST", body: fd })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.error || `${type} audit failed`)
      }
      const data = await res.json()
      setAdvancedResults((prev) => ({ ...prev, [type]: data }))
    } catch (error) {
      setAdvancedResults((prev) => ({ ...prev, [type]: { error: error instanceof Error ? error.message : "Request failed" } }))
    } finally {
      setIsAdvancedLoading((prev) => ({ ...prev, [type]: false }))
    }
  }

  const downloadReport = () => {
    if (!analysisResults) return
    const r = analysisResults
    const lines: string[] = []
    const add = (s: string) => lines.push(s)

    add("AI SHAYAK - ETHICAL AI AUDIT REPORT")
    add("====================================")
    add("")
    add("Generated: " + new Date().toISOString())
    add("Model: " + (uploadedModel?.name || "N/A"))
    add("Dataset: " + (uploadedDataset?.name || "N/A"))
    add("Sensitive Attribute: " + sensitiveColumn)
    add("Framework: " + (r.framework || "N/A"))
    add("")
    add("OVERALL STATUS")
    add("--------------")
    add("Ethics Score: " + r.ethicsScore + "/10")
    add("Risk Level: " + r.riskLevel)
    add("Fairness Score: " + r.fairnessScore + "/10")
    add("Integrity Score: " + r.integrityScore + "/10")
    add("Transparency Score: " + r.transparencyScore + "/10")
    add("")
    add("FAIRNESS METRICS")
    add("----------------")
    add("Demographic Parity Difference: " + r.metrics.demographicParityDifference + " (ideal: 0)")
    add("Equalized Odds Difference: " + r.metrics.equalizedOddsDifference + " (ideal: 0)")
    add("Disparate Impact Ratio: " + r.metrics.disparateImpact + " (ideal: 1.0, threshold: 0.8)")
    add("Robustness: " + (r.metrics.robustness * 100).toFixed(1) + "%")
    add("Privacy: " + (r.metrics.privacy * 100).toFixed(1) + "%")
    add("")

    if (r.groupPerformance) {
      add("GROUP PERFORMANCE")
      add("-----------------")
      Object.entries(r.groupPerformance).forEach(([g, d]: [string, any]) => {
        add("  " + g + ": Selection Rate=" + (d.selectionRate?.toFixed(3) || "N/A") + ", Acc=" + ((d.accuracy || 0) * 100).toFixed(1) + "%, Count=" + d.count)
      })
      add("")
    }
    add("RECOMMENDATIONS")
    add("---------------")
    ;(r.recommendations || []).forEach((rec: string, i: number) => add((i + 1) + ". " + rec))
    add("")
    add("COMPLIANCE")
    add("----------")
    ;(r.compliance?.frameworks || []).forEach((fw: any) => {
      add(fw.name + ": " + fw.status)
      add("  " + fw.requirement + " - " + fw.details)
    })
    add("")

    Object.entries(advancedResults).forEach(([key, val]: [string, any]) => {
      if (val.error) return
      add(key.toUpperCase() + " ANALYSIS")
      add("=".repeat(key.length + 10))
      if (val.recommendation) add("Recommended: " + val.recommendation)
      if (val.interpretation) add("Interpretation: " + val.interpretation)
      if (val.riskLevel) add("Risk Level: " + val.riskLevel)
      if (val.recommended) add("Optimal Threshold: " + val.recommended.threshold + " (SPD=" + val.recommended.spd + ", Acc=" + val.recommended.accuracy + ")")
      if (val.potentialBiasedFeatures?.length) add("Biased Features: " + val.potentialBiasedFeatures.join(", "))
      if (val.accuracyUnderAttack !== undefined) add("Accuracy Under Attack: " + (val.accuracyUnderAttack * 100).toFixed(1) + "%")
      if (val.attackSuccessRate !== undefined) add("Attack Success Rate: " + (val.attackSuccessRate * 100).toFixed(1) + "%")
      if (val.riskScore !== undefined) add("Membership Inference Risk Score: " + val.riskScore)
      if (val.estimatedEpsilon !== undefined) add("Estimated DP Epsilon: " + val.estimatedEpsilon)
      add("")
    })

    const blob = new Blob([lines.join("\\n")], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "AI_Shayak_Audit_" + new Date().toISOString().split("T")[0] + ".txt"
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const DR = () => analysisResults || {
    ethicsScore: 0, fairnessScore: 0, integrityScore: 0, transparencyScore: 0,
    biasDetected: false, riskLevel: "Unknown", framework: "N/A",
    metrics: { demographicParityDifference: 0, equalizedOddsDifference: 0, disparateImpact: 0, robustness: 0, privacy: 0 },
    compliance: { frameworks: [], modelCard: { intendedUse: "", limitations: "", fairnessPhilosophy: "" } },
    recommendations: [], chartData: [], groupPerformance: {},
  }

  const chartCfg = { value: { label: "Value", color: "hsl(var(--primary))" } }

  const dropZone = (type: string) =>
    "border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 cursor-pointer " +
    (dragOver === type ? "border-primary bg-primary/10 scale-[1.01]" : "border-border hover:border-primary hover:bg-card/30")

  const renderAdvancedResult = (type: string, data: any) => {
    if (!data) return null
    if (data.error) return <Alert variant="destructive"><AlertTitle>Error</AlertTitle><AlertDescription>{data.error}</AlertDescription></Alert>

    if (type === "threshold") {
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="bg-card/30 border-primary"><CardHeader className="pb-2"><CardTitle className="text-sm">Optimal Fairness</CardTitle></CardHeader>
              <CardContent><p className="text-2xl font-bold">{data.optimalFairness?.threshold}</p><p className="text-xs text-gray-400">SPD: {data.optimalFairness?.spd}</p></CardContent></Card>
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">Best Accuracy</CardTitle></CardHeader>
              <CardContent><p className="text-2xl font-bold">{data.optimalAccuracy?.threshold}</p><p className="text-xs text-gray-400">Acc: {(data.optimalAccuracy?.accuracy * 100).toFixed(1)}%</p></CardContent></Card>
            <Card className="bg-card/30 border-green-500"><CardHeader className="pb-2"><CardTitle className="text-sm">Recommended</CardTitle></CardHeader>
              <CardContent><p className="text-2xl font-bold text-green-400">{data.recommended?.threshold}</p><p className="text-xs text-gray-400">SPD: {data.recommended?.spd} Acc: {(data.recommended?.accuracy * 100).toFixed(1)}%</p></CardContent></Card>
          </div>
          <p className="text-primary font-medium">{data.recommendation}</p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm"><thead className="bg-muted/50">
              <tr><th className="p-2 text-left">Threshold</th><th className="p-2 text-left">SPD</th><th className="p-2 text-left">DI</th><th className="p-2 text-left">Accuracy</th><th className="p-2 text-left">Bias</th></tr>
            </thead><tbody>
              {(data.thresholds || []).map((t: any, i: number) => (
                <tr key={i} className={"border-t border-border " + (t.threshold === data.recommended?.threshold ? "bg-primary/10" : "")}>
                  <td className="p-2 font-mono">{t.threshold}</td><td className="p-2 font-mono">{t.spd}</td><td className="p-2 font-mono">{t.di}</td>
                  <td className="p-2 font-mono">{(t.accuracy * 100).toFixed(1)}%</td>
                  <td className="p-2">{t.biasDetected ? <Badge variant="destructive">Bias</Badge> : <Badge variant="default">Fair</Badge>}</td>
                </tr>
              ))}
            </tbody></table>
          </div>
        </div>
      )
    }

    if (type === "retraining") {
      return (
        <div className="space-y-4">
          {(data.potentialBiasedFeatures?.length > 0) && (
            <Alert><AlertTitle>Biased Features</AlertTitle><AlertDescription>{data.potentialBiasedFeatures.join(", ")}</AlertDescription></Alert>
          )}
          <Card className="bg-card/30"><CardContent className="pt-6 space-y-4">
            <p><strong>Resampling:</strong> {data.resamplingSuggestion}</p>
            <p><strong>Reweighting:</strong> {data.reweightingSuggestion}</p>
            {data.adversarialDebiasingSuggestion && <p><strong>Advanced:</strong> {data.adversarialDebiasingSuggestion}</p>}
          </CardContent></Card>
          <div className="overflow-x-auto">
            <table className="w-full text-sm"><thead className="bg-muted/50">
              <tr><th className="p-2 text-left">Feature</th><th className="p-2 text-left">Correlation w/ Sensitive</th></tr>
            </thead><tbody>
              {(data.featureCorrelations || []).map((f: any, i: number) => (
                <tr key={i} className="border-t border-border">
                  <td className="p-2 font-mono">{f.feature}</td>
                  <td className="p-2"><Progress value={f.correlation_with_sensitive * 100} className="h-2 w-32 inline-block" /> <span className="ml-2 text-xs">{f.correlation_with_sensitive}</span></td>
                </tr>
              ))}
            </tbody></table>
          </div>
        </div>
      )
    }

    if (type === "intersectional") {
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">SPD</CardTitle></CardHeader>
              <CardContent><p className="text-xl font-bold">{data.fairness?.spd}</p></CardContent></Card>
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">DI</CardTitle></CardHeader>
              <CardContent><p className="text-xl font-bold">{data.fairness?.di}</p></CardContent></Card>
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">EOD</CardTitle></CardHeader>
              <CardContent><p className="text-xl font-bold">{data.fairness?.eod}</p></CardContent></Card>
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">AOD</CardTitle></CardHeader>
              <CardContent><p className="text-xl font-bold">{data.fairness?.aod}</p></CardContent></Card>
          </div>
          <div className="h-[250px]">
            <ChartContainer config={chartCfg} className="h-full w-full">
              <BarChart data={data.chartData || []}>
                <XAxis dataKey="group" stroke="#888" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="#888" fontSize={11} tickLine={false} axisLine={false} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="value" radius={[4,4,0,0]}>
                  {(data.chartData || []).map((_: any, i: number) => <Cell key={i} fill={i%2===0 ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.6)"} />)}
                </Bar>
              </BarChart>
            </ChartContainer>
          </div>
        </div>
      )
    }

    if (type === "calibration") {
      return (
        <div className="space-y-4">
          <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">Calibration Error</CardTitle></CardHeader>
            <CardContent><p className="text-3xl font-bold">{data.calibrationError}</p><p className="text-xs text-gray-400">Lower is better</p></CardContent></Card>
          {data.calibrationData && Object.entries(data.calibrationData).map(([group, bins]: [string, any]) => (
            <div key={group} className="p-4 bg-muted/20 rounded-2xl">
              <h4 className="font-semibold mb-2">{group}</h4>
              <table className="w-full text-sm"><thead className="bg-muted/50">
                <tr><th className="p-2 text-left">Bin</th><th className="p-2 text-left">Mean Pred</th><th className="p-2 text-left">Mean Actual</th><th className="p-2 text-left">Count</th></tr>
              </thead><tbody>
                {Object.entries(bins).map(([bin, v]: [string, any]) => (
                  <tr key={bin} className="border-t border-border">
                    <td className="p-2 font-mono">{bin}</td><td className="p-2">{v.meanPred?.toFixed(3)}</td><td className="p-2">{v.meanActual?.toFixed(3)}</td><td className="p-2">{v.count}</td>
                  </tr>
                ))}
              </tbody></table>
            </div>
          ))}
        </div>
      )
    }

    if (type === "adversarial") {
      return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="bg-card/30 border-green-500"><CardHeader className="pb-2"><CardTitle className="text-sm">Accuracy Under Attack</CardTitle></CardHeader>
            <CardContent><p className="text-3xl font-bold text-green-400">{(data.accuracyUnderAttack * 100).toFixed(1)}%</p></CardContent></Card>
          <Card className="bg-card/30 border-red-500"><CardHeader className="pb-2"><CardTitle className="text-sm">Attack Success Rate</CardTitle></CardHeader>
            <CardContent><p className="text-3xl font-bold text-red-400">{(data.attackSuccessRate * 100).toFixed(1)}%</p></CardContent></Card>
          <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">Perturbation</CardTitle></CardHeader>
            <CardContent><p className="text-xl font-bold">{data.perturbationEpsilon}</p><p className="text-xs text-gray-400">{data.perturbationDescription}</p></CardContent></Card>
        </div>
      )
    }

    if (type === "differential-privacy") {
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">DP Epsilon</CardTitle></CardHeader>
              <CardContent><p className="text-3xl font-bold">{data.estimatedEpsilon}</p></CardContent></Card>
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">Max Influence</CardTitle></CardHeader>
              <CardContent><p className="text-xl font-bold">{data.maxInfluence}</p></CardContent></Card>
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">Risk Level</CardTitle></CardHeader>
              <CardContent><Badge variant={data.riskLevel === "High" ? "destructive" : data.riskLevel === "Medium" ? "outline" : "default"}>{data.riskLevel}</Badge></CardContent></Card>
          </div>
          <p className="text-sm text-gray-400">{data.interpretation}</p>
        </div>
      )
    }

    if (type === "membership-inference") {
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">Member Confidence</CardTitle></CardHeader>
              <CardContent><p className="text-xl font-bold">{(data.memberConfidence * 100).toFixed(1)}%</p></CardContent></Card>
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">Non-Member Confidence</CardTitle></CardHeader>
              <CardContent><p className="text-xl font-bold">{(data.nonMemberConfidence * 100).toFixed(1)}%</p></CardContent></Card>
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">Risk Score</CardTitle></CardHeader>
              <CardContent><p className={"text-2xl font-bold " + (data.riskScore > 0.1 ? "text-red-400" : "text-green-400")}>{data.riskScore}</p></CardContent></Card>
            <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">Risk Level</CardTitle></CardHeader>
              <CardContent><Badge variant={data.riskLevel === "High" ? "destructive" : data.riskLevel === "Medium" ? "outline" : "default"}>{data.riskLevel}</Badge></CardContent></Card>
          </div>
          <p className="text-sm text-gray-400">{data.interpretation}</p>
        </div>
      )
    }

    if (type === "regression-fairness") {
      return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">Mean Prediction Diff</CardTitle></CardHeader>
            <CardContent><p className="text-2xl font-bold">{data.meanPredictionDifference}</p></CardContent></Card>
          <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">MAE Disparity</CardTitle></CardHeader>
            <CardContent><p className="text-2xl font-bold">{data.maxMaeDisparity}</p></CardContent></Card>
          <Card className="bg-card/30"><CardHeader className="pb-2"><CardTitle className="text-sm">Overall Mean</CardTitle></CardHeader>
            <CardContent><p className="text-2xl font-bold">{data.overallMean}</p></CardContent></Card>
        </div>
      )
    }

    return <pre className="text-xs text-gray-400 overflow-auto max-h-96">{JSON.stringify(data, null, 2)}</pre>
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
              <p className="text-sm text-gray-300">Ethical AI Governance Platform</p>
            </div>
          </div>
        </div>
      </header>
      <main className="w-full flex flex-col items-center py-12 space-y-12">
        <div className="text-center space-y-6 mb-12 px-8">
          <h2 className="text-5xl font-bold">AI Ethics & Bias Audit</h2>
          <p className="text-xl text-gray-300 max-w-5xl mx-auto leading-relaxed">
            Multi-framework fairness auditing, security analysis, threshold optimization, deep fairness metrics, and regulatory compliance for ML models.
          </p>
        </div>
        <div className="w-full flex flex-col items-center space-y-10">
          <Card className="border-2 border-border bg-card/50 backdrop-blur-sm w-[95%] max-w-[1600px]">
            <CardHeader className="pb-6">
              <CardTitle className="flex items-center space-x-3 text-2xl">
                <Upload className="w-6 h-6" />
                <span>Input Assets</span>
              </CardTitle>
              <CardDescription className="text-base text-gray-300">
                Upload ML artifacts for evaluation. Supports scikit-learn, XGBoost, PyTorch, and TensorFlow.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-8">
              <div className="space-y-4">
                <Label className="text-lg font-medium">Audit Scope</Label>
                <RadioGroup value={evaluationType} onValueChange={(v: "model" | "dataset") => setEvaluationType(v)}>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="model" id="m" /><Label htmlFor="m" className="text-base">End-to-End Audit</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="dataset" id="d" /><Label htmlFor="d" className="text-base">Data-Only Ethics Audit</Label>
                  </div>
                </RadioGroup>
              </div>
              <Separator />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className={dropZone("model")}
                  onDragOver={(e) => { e.preventDefault(); setDragOver("model") }}
                  onDragLeave={() => setDragOver(null)}
                  onDrop={(e) => handleDrop("model", e)}
                  onClick={() => modelInputRef.current?.click()}>
                  <Brain className="w-10 h-10 text-gray-400 mx-auto mb-4" />
                  <p className="text-lg font-medium mb-2">Model Artifact</p>
                  <p className="text-sm text-gray-400 mb-4">Drop or click to upload</p>
                  <p className="text-xs text-gray-500">.pkl .joblib .json .ubj .pt .pth .h5 .keras</p>
                  <input ref={modelInputRef} type="file" accept={MODEL_EXTS} onChange={handleModelUpload} className="hidden" disabled={evaluationType === "dataset"} />
                  {uploadedModel && evaluationType !== "dataset" && <Badge variant="outline" className="mt-3">{uploadedModel.name}</Badge>}
                </div>
                <div className={dropZone("dataset")}
                  onDragOver={(e) => { e.preventDefault(); setDragOver("dataset") }}
                  onDragLeave={() => setDragOver(null)}
                  onDrop={(e) => handleDrop("dataset", e)}
                  onClick={() => datasetInputRef.current?.click()}>
                  <FileText className="w-10 h-10 text-gray-400 mx-auto mb-4" />
                  <p className="text-lg font-medium mb-2">Dataset Samples</p>
                  <p className="text-sm text-gray-400 mb-4">Drop or click to upload</p>
                  <p className="text-xs text-gray-500">CSV format with target column</p>
                  <input ref={datasetInputRef} type="file" accept=".csv" onChange={handleDatasetUpload} className="hidden" />
                  {uploadedDataset && <Badge variant="outline" className="mt-3">{uploadedDataset.name}</Badge>}
                </div>
              </div>
              {uploadedDataset && !showSchema && (
                <Button variant="ghost" size="sm" onClick={loadSchemaPreview} className="flex items-center gap-2">
                  <Table2 className="w-4 h-4" /> Preview Schema
                </Button>
              )}
              {showSchema && schemaPreview && (
                <Card className="bg-muted/20 border-border">
                  <CardHeader className="pb-2"><CardTitle className="text-lg flex items-center gap-2"><Table2 className="w-5 h-5" /> Dataset Schema</CardTitle>
                    <CardDescription>{schemaPreview.rowCount} rows x {schemaPreview.columnCount} columns</CardDescription></CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto max-h-64 overflow-y-auto">
                      <table className="w-full text-sm"><thead className="bg-muted/50">
                        <tr><th className="p-2 text-left">Column</th><th className="p-2 text-left">Type</th><th className="p-2 text-left">Unique</th><th className="p-2 text-left">Missing</th><th className="p-2 text-left">Sample</th></tr>
                      </thead><tbody>
                        {schemaPreview.columns?.map((col: any, i: number) => (
                          <tr key={i} className="border-t border-border">
                            <td className={"p-2 font-mono " + (col.name === sensitiveColumn ? "text-primary font-bold" : "")}>{col.name}</td>
                            <td className="p-2 text-xs">{col.dtype}</td>
                            <td className="p-2">{col.uniqueCount}</td>
                            <td className="p-2">{col.missingCount}</td>
                            <td className="p-2 text-xs truncate max-w-[200px]">{(col.sampleValues || []).slice(0, 3).join(", ") || "-"}</td>
                          </tr>
                        ))}
                      </tbody></table>
                    </div>
                  </CardContent>
                </Card>
              )}
              {schemaLoading && <Progress value={50} className="h-1" />}
              {(uploadedModel || uploadedDataset) && (
                <div className="space-y-6 p-8 bg-muted/30 rounded-2xl">
                  <div className="space-y-4">
                    <Label htmlFor="sc" className="text-lg font-medium">Sensitive Attribute</Label>
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <Input id="sc" placeholder="e.g., race, gender, age" value={sensitiveColumn} onChange={(e) => setSensitiveColumn(e.target.value)} className="pl-10 text-lg h-12" list="col-suggestions" />
                    </div>
                    <datalist id="col-suggestions">{availableColumns.map((c) => (<option key={c} value={c} />))}</datalist>
                    <p className="text-sm text-gray-400">Protected demographic group column for fairness analysis.</p>
                  </div>
                  <Button onClick={startAnalysis} size="lg" className="w-full h-12 text-lg font-medium" disabled={isAnalyzing}>
                    {isAnalyzing ? "Running Audit..." : "Run Ethics Evaluation"}
                  </Button>
                </div>
              )}
              {errorMessage && (
                <Alert variant="destructive"><AlertTriangle className="h-5 w-5" /><AlertTitle>Audit Failed</AlertTitle><AlertDescription>{errorMessage}</AlertDescription></Alert>
              )}
              {isAnalyzing && <Progress value={analysisProgress} className="w-full h-3" />}
            </CardContent>
          </Card>
          {analysisComplete && (
            <div className="w-[95%] max-w-[1600px] space-y-8">
              <div className="flex justify-between items-center">
                <h3 className="text-3xl font-bold">Audit Results</h3>
                <div className="flex items-center gap-4">
                  {analysisResults?.framework && (
                    <Badge variant="outline" className="text-sm flex items-center gap-1">
                      <Brain className="w-4 h-4" /> {analysisResults.framework}
                    </Badge>
                  )}
                  <Button onClick={downloadReport} variant="outline" className="flex items-center space-x-2">
                    <Download className="w-5 h-5" /><span>Download Full Report</span>
                  </Button>
                </div>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <Card className="bg-card/50 border-2 border-primary">
                  <CardHeader className="pb-2"><CardTitle className="text-lg text-gray-300">Overall Ethics Score</CardTitle></CardHeader>
                  <CardContent>
                    <div className="text-5xl font-bold">{DR().ethicsScore}/10</div>
                    <Badge className="mt-2" variant={DR().riskLevel === "Low" ? "default" : DR().riskLevel === "Medium" ? "outline" : "destructive"}>{DR().riskLevel} Risk</Badge>
                  </CardContent>
                </Card>
                <Card className="bg-card/30 border-2 border-border">
                  <CardHeader className="pb-2"><CardTitle className="text-lg text-gray-300">Fairness</CardTitle></CardHeader>
                  <CardContent><div className="text-4xl font-bold">{DR().fairnessScore}/10</div><Progress value={DR().fairnessScore * 10} className="mt-3 h-2" /></CardContent>
                </Card>
                <Card className="bg-card/30 border-2 border-border">
                  <CardHeader className="pb-2"><CardTitle className="text-lg text-gray-300">Data Integrity</CardTitle></CardHeader>
                  <CardContent><div className="text-4xl font-bold">{DR().integrityScore}/10</div><Progress value={DR().integrityScore * 10} className="mt-3 h-2" /></CardContent>
                </Card>
                <Card className="bg-card/30 border-2 border-border">
                  <CardHeader className="pb-2"><CardTitle className="text-lg text-gray-300">Transparency</CardTitle></CardHeader>
                  <CardContent><div className="text-4xl font-bold">{DR().transparencyScore}/10</div><Progress value={DR().transparencyScore * 10} className="mt-3 h-2" /></CardContent>
                </Card>
              </div>
              <Card className="border-2 border-border bg-card/50 backdrop-blur-sm">
                <CardContent className="p-8">
                  <Tabs defaultValue="bias" className="w-full">
                    <TabsList className="grid w-full grid-cols-5 h-14">
                      <TabsTrigger value="bias" className="text-lg"><AlertTriangle className="w-5 h-5 mr-2" /> Bias & Fairness</TabsTrigger>
                      <TabsTrigger value="security" className="text-lg"><ShieldCheck className="w-5 h-5 mr-2" /> Security Audit</TabsTrigger>
                      <TabsTrigger value="groups" className="text-lg"><Users className="w-5 h-5 mr-2" /> Group Analysis</TabsTrigger>
                      <TabsTrigger value="compliance" className="text-lg"><FileText className="w-5 h-5 mr-2" /> Compliance</TabsTrigger>
                      <TabsTrigger value="recommendations" className="text-lg"><CheckCircle className="w-5 h-5 mr-2" /> Recommendations</TabsTrigger>
                    </TabsList>
                    <TabsContent value="bias" className="pt-8 space-y-8">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-6">
                          <div className="p-6 bg-muted/20 rounded-2xl border border-border">
                            <h4 className="text-xl font-semibold mb-4">Parity Metrics</h4>
                            <div className="space-y-4">
                              <div className="flex justify-between items-center"><span className="text-gray-300">Demographic Parity Diff</span><span className="font-mono text-xl">{DR().metrics.demographicParityDifference}</span></div>
                              <Separator className="bg-border/50" />
                              <div className="flex justify-between items-center"><span className="text-gray-300">Equalized Odds Diff</span><span className="font-mono text-xl">{DR().metrics.equalizedOddsDifference}</span></div>
                            </div>
                          </div>
                        </div>
                        <div className="space-y-6">
                          <div className="p-6 bg-muted/20 rounded-2xl border border-border">
                            <h4 className="text-xl font-semibold mb-4">Impact Ratios</h4>
                            <div className="space-y-4">
                              <div className="flex justify-between items-center">
                                <span className="text-gray-300">Disparate Impact Ratio</span>
                                <span className={"font-mono text-xl " + (DR().metrics.disparateImpact < 0.8 ? "text-red-400" : "text-green-400")}>{DR().metrics.disparateImpact}</span>
                              </div>
                              <Separator className="bg-border/50" />
                              <div className="flex justify-between items-center"><span className="text-gray-300">Mitigation Strategy</span><Badge variant="outline" className="text-primary border-primary">Post-Processing</Badge></div>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="p-6 bg-primary/10 rounded-2xl border border-primary/30">
                        <h4 className="text-xl font-semibold mb-3 flex items-center"><Brain className="w-6 h-6 mr-2 text-primary" /> Mitigation Plan</h4>
                        <p className="text-gray-300 text-lg">{DR().recommendations[0] || "No critical bias detected."}</p>
                      </div>
                    </TabsContent>
                    <TabsContent value="security" className="pt-8 space-y-8">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <Card className="bg-card/30 border-border">
                          <CardHeader><CardTitle className="flex items-center"><ShieldCheck className="w-5 h-5 mr-2 text-blue-400" /> Robustness</CardTitle><CardDescription>Sensitivity to input perturbations</CardDescription></CardHeader>
                          <CardContent>
                            <div className="text-4xl font-bold mb-2">{(DR().metrics.robustness * 100).toFixed(1)}%</div>
                            <p className="text-sm text-gray-400">Stability under +/-5% Gaussian noise.</p>
                            <Progress value={DR().metrics.robustness * 100} className="mt-4 h-2" />
                          </CardContent>
                        </Card>
                        <Card className="bg-card/30 border-border">
                          <CardHeader><CardTitle className="flex items-center"><Search className="w-5 h-5 mr-2 text-purple-400" /> Privacy Leakage</CardTitle><CardDescription>Attribute inference risk</CardDescription></CardHeader>
                          <CardContent>
                            <div className="text-4xl font-bold mb-2">{(DR().metrics.privacy * 100).toFixed(1)}%</div>
                            <p className="text-sm text-gray-400">Privacy score (higher = less leakage).</p>
                            <Progress value={DR().metrics.privacy * 100} className="mt-4 h-2" />
                          </CardContent>
                        </Card>
                      </div>
                    </TabsContent>
                    <TabsContent value="compliance" className="pt-8 space-y-6">
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {DR().compliance?.frameworks?.map((fw: any, idx: number) => (
                          <div key={idx} className="p-6 bg-muted/20 rounded-2xl border border-border">
                            <div className="flex justify-between items-start mb-4">
                              <h4 className="text-xl font-bold">{fw.name}</h4>
                              <Badge variant={fw.status === "Compliant" ? "default" : "destructive"}>{fw.status}</Badge>
                            </div>
                            <p className="text-primary text-sm font-semibold mb-2">{fw.requirement}</p>
                            <p className="text-gray-400">{fw.details}</p>
                          </div>
                        ))}
                      </div>
                      <Card className="bg-card/30 border-dashed border-2 border-border">
                        <CardHeader><CardTitle>Generated Model Card</CardTitle></CardHeader>
                        <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-6">
                          <div><h5 className="text-sm font-bold text-gray-500 uppercase mb-2">Intended Use</h5><p className="text-gray-300">{DR().compliance?.modelCard?.intendedUse}</p></div>
                          <div><h5 className="text-sm font-bold text-gray-500 uppercase mb-2">Fairness Philosophy</h5><p className="text-gray-300">{DR().compliance?.modelCard?.fairnessPhilosophy}</p></div>
                          <div><h5 className="text-sm font-bold text-gray-500 uppercase mb-2">Limitations</h5><p className="text-gray-300">{DR().compliance?.modelCard?.limitations}</p></div>
                        </CardContent>
                      </Card>
                    </TabsContent>
                    <TabsContent value="groups" className="pt-8">
                      <div className="grid grid-cols-1 xl:grid-cols-2 gap-10">
                        <div className="h-[350px]">
                          <h4 className="text-xl font-semibold mb-6 text-center">{evaluationType === "model" ? "Selection Rate by Group" : "Dataset Distribution"}</h4>
                          <ChartContainer config={chartCfg} className="h-full w-full">
                            <BarChart data={DR().chartData}>
                              <XAxis dataKey="group" stroke="#888" fontSize={12} tickLine={false} axisLine={false} />
                              <YAxis stroke="#888" fontSize={12} tickLine={false} axisLine={false} />
                              <ChartTooltip content={<ChartTooltipContent />} />
                              <Bar dataKey="value" radius={[4,4,0,0]}>
                                {DR().chartData.map((_: any, i: number) => <Cell key={i} fill={i%2===0 ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.6)"} />)}
                              </Bar>
                            </BarChart>
                          </ChartContainer>
                        </div>
                        <div className="overflow-hidden rounded-2xl border border-border">
                          <table className="w-full text-left">
                            <thead className="bg-muted/50">
                              <tr><th className="p-4 font-semibold">Group</th><th className="p-4 font-semibold">{evaluationType === "model" ? "Selection Rate" : "Count"}</th><th className="p-4 font-semibold">{evaluationType === "model" ? "Accuracy" : "Proportion"}</th></tr>
                            </thead>
                            <tbody>
                              {Object.entries(DR().groupPerformance).map(([g, d]: [string, any]) => (
                                <tr key={g} className="border-t border-border hover:bg-muted/20">
                                  <td className="p-4">{g}</td>
                                  <td className="p-4 font-mono">{evaluationType === "model" ? d.selectionRate?.toFixed(3) : d.count}</td>
                                  <td className="p-4 font-mono">{evaluationType === "model" ? (d.accuracy * 100).toFixed(1) + "%" : (d.percentage * 100).toFixed(1) + "%"}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </TabsContent>
                    <TabsContent value="recommendations" className="pt-8">
                      <div className="space-y-4">
                        {(DR().recommendations || []).map((rec: string, i: number) => (
                          <div key={i} className="flex items-start p-4 bg-muted/20 rounded-xl border-l-4 border-primary">
                            <CheckCircle className="w-6 h-6 mr-4 text-primary shrink-0" /><span className="text-lg text-gray-200">{rec}</span>
                          </div>
                        ))}
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
              <Card className="border-2 border-border bg-card/50 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-3 text-2xl">
                    <Gauge className="w-6 h-6" />
                    <span>Advanced Analysis</span>
                  </CardTitle>
                  <CardDescription className="text-base text-gray-300">
                    Threshold optimization, retraining suggestions, deep fairness metrics, and security vulnerability assessment.
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-8">
                  <Tabs value={activeAdvancedTab} onValueChange={setActiveAdvancedTab} className="w-full">
                    <TabsList className="grid w-full grid-cols-3 h-14">
                      <TabsTrigger value="optimization" className="text-lg"><Sliders className="w-5 h-5 mr-2" /> Optimization</TabsTrigger>
                      <TabsTrigger value="deep-fairness" className="text-lg"><Layers className="w-5 h-5 mr-2" /> Deep Fairness</TabsTrigger>
                      <TabsTrigger value="security-advanced" className="text-lg"><Lock className="w-5 h-5 mr-2" /> Security</TabsTrigger>
                    </TabsList>
                    <TabsContent value="optimization" className="pt-8 space-y-6">
                      <div className="flex flex-wrap gap-4">
                        <Button variant="outline" size="sm" onClick={() => runAdvancedAudit("threshold")} disabled={isAdvancedLoading["threshold"]} className="flex items-center gap-2">
                          {isAdvancedLoading["threshold"] ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Target className="w-4 h-4" />} Tune Thresholds
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => runAdvancedAudit("retraining")} disabled={isAdvancedLoading["retraining"]} className="flex items-center gap-2">
                          {isAdvancedLoading["retraining"] ? <RefreshCw className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />} Retraining Suggestions
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => runAdvancedAudit("regression-fairness")} disabled={isAdvancedLoading["regression-fairness"]} className="flex items-center gap-2">
                          {isAdvancedLoading["regression-fairness"] ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ArrowUpDown className="w-4 h-4" />} Regression Fairness
                        </Button>
                      </div>
                      {renderAdvancedResult("threshold", advancedResults.threshold)}
                      {renderAdvancedResult("retraining", advancedResults.retraining)}
                      {renderAdvancedResult("regression-fairness", advancedResults["regression-fairness"])}
                    </TabsContent>
                    <TabsContent value="deep-fairness" className="pt-8 space-y-6">
                      <div className="flex flex-col md:flex-row gap-4 items-start md:items-center">
                        <div className="flex gap-2">
                          <Button variant={deepAuditType === "intersectional" ? "default" : "outline"} size="sm" onClick={() => setDeepAuditType("intersectional")}><Layers className="w-4 h-4 mr-2" /> Intersectional</Button>
                          <Button variant={deepAuditType === "calibration" ? "default" : "outline"} size="sm" onClick={() => setDeepAuditType("calibration")}><BarChart3 className="w-4 h-4 mr-2" /> Calibration</Button>
                        </div>
                        {deepAuditType === "intersectional" && (
                          <div className="flex items-center gap-2">
                            <Label className="text-sm">2nd Column:</Label>
                            <Input placeholder="e.g., race" value={sensitiveColumnsMulti} onChange={(e) => setSensitiveColumnsMulti(e.target.value)} className="h-9 w-40" list="col-suggestions" />
                          </div>
                        )}
                        <Button variant="default" size="sm" onClick={() => runAdvancedAudit(deepAuditType)} disabled={isAdvancedLoading[deepAuditType]} className="flex items-center gap-2">
                          {isAdvancedLoading[deepAuditType] ? <RefreshCw className="w-4 h-4 animate-spin" /> : <PlayIcon />} Run {deepAuditType}
                        </Button>
                      </div>
                      {renderAdvancedResult("intersectional", advancedResults.intersectional)}
                      {renderAdvancedResult("calibration", advancedResults.calibration)}
                    </TabsContent>
                    <TabsContent value="security-advanced" className="pt-8 space-y-6">
                      <div className="flex flex-wrap gap-4">
                        <Button variant="outline" size="sm" onClick={() => runAdvancedAudit("adversarial")} disabled={isAdvancedLoading["adversarial"]} className="flex items-center gap-2">
                          {isAdvancedLoading["adversarial"] ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ShieldCheck className="w-4 h-4" />} Adversarial Robustness
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => runAdvancedAudit("differential-privacy")} disabled={isAdvancedLoading["differential-privacy"]} className="flex items-center gap-2">
                          {isAdvancedLoading["differential-privacy"] ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Lock className="w-4 h-4" />} Differential Privacy
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => runAdvancedAudit("membership-inference")} disabled={isAdvancedLoading["membership-inference"]} className="flex items-center gap-2">
                          {isAdvancedLoading["membership-inference"] ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />} Membership Inference
                        </Button>
                      </div>
                      {renderAdvancedResult("adversarial", advancedResults.adversarial)}
                      {renderAdvancedResult("differential-privacy", advancedResults["differential-privacy"])}
                      {renderAdvancedResult("membership-inference", advancedResults["membership-inference"])}
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
          <p>&copy; 2026 AI Shayak - Empowering Ethical Machine Learning</p>
        </div>
      </footer>
    </div>
  )
}

function PlayIcon(props: any) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
  )
}
