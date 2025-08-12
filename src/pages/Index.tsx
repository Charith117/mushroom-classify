import { useState, useRef } from "react";
import { Navigation } from "@/components/Navigation";
import { Footer } from "@/components/Footer";
import { MushroomCard } from "@/components/MushroomCard";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Link } from "react-router-dom";
import { Camera, Leaf, Users, Sparkles, Upload, Thermometer, Droplets, Sun } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import heroImage from "@/assets/hero-mushrooms.jpg";
import shiitakeImage from "@/assets/shiitake.jpg";
import oysterImage from "@/assets/oyster.jpg";
import buttonImage from "@/assets/button.jpg";
import portobelloImage from "@/assets/portobello.jpg";

interface PredictionResult {
  mushroomType?: {
    common: string;
    scientific: string;
  };
  confidence?: number;
  classId?: number;
  features?: {
    colorMean: [number, number, number];
    textureFeatures: {
      contrast: number;
      correlation: number;
      energy: number;
      homogeneity: number;
      entropy: number;
    };
  };
  growingParameters?: {
    spawnRun: { temperature: string; humidity: string; co2: string; light: string };
    primordiaFormation: { temperature: string; humidity: string; co2: string; light: string };
    fruitingBodyFormation: { temperature: string; humidity: string; co2: string; light: string };
  };
}

const Index = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataURL = e.target?.result as string;
        // Extract just the base64 string from the data URL
        const base64String = dataURL.split(',')[1];
        setUploadedImage(base64String);
      };
      reader.readAsDataURL(file);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const analyzeMushroom = async () => {
    if (!uploadedImage) {
      toast({
        title: "Missing Information",
        description: "Please upload an image first.",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    
    try {
      // Debug: Log what we're sending
      console.log('Image data length:', uploadedImage.length);
      console.log('Image data preview:', uploadedImage.substring(0, 100) + '...');
      
      // Call your ML backend API
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: uploadedImage
        })
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`Failed to analyze image: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      
      // Check if the response contains an error
      if (result.error) {
        // Handle error responses from the backend
        toast({
          title: result.error,
          description: result.message || result.suggestion || "Please try with a different image.",
          variant: "destructive",
        });
        return;
      }
      
      // Check if required fields exist
      if (!result.mushroomType || !result.confidence || result.classId === undefined) {
        throw new Error('Invalid response format from server');
      }
      
      const predictionResult: PredictionResult = {
        mushroomType: result.mushroomType,
        confidence: result.confidence,
        classId: result.classId,
        features: result.features,
        growingParameters: getGrowingParameters(result.classId)
      };
      
      setPrediction(predictionResult);
      
      toast({
        title: "Analysis Complete!",
        description: `Identified as ${predictionResult.mushroomType?.common} with ${predictionResult.confidence}% confidence.`,
      });
      
    } catch (error) {
      console.error('Analysis error:', error);
      toast({
        title: "Analysis Failed",
        description: "Could not analyze the image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getGrowingParameters = (classId: number) => {
    const params: Record<number, any> = {
      0: { // Abalone
        spawnRun: { temperature: "24-30°C", humidity: "90-95%", co2: "5,000-20,000 ppm", light: "Total dark" },
        primordiaFormation: { temperature: "18-24°C", humidity: "95-100%", co2: "500-1,000 ppm", light: "500-1,000 lux" },
        fruitingBodyFormation: { temperature: "21-27°C", humidity: "85-90%", co2: "<2,000 ppm", light: "500-1,000 lux" }
      },
      1: { // Pink Oyster
        spawnRun: { temperature: "24-27°C", humidity: "90-100%", co2: "20,000 ppm", light: "Total dark" },
        primordiaFormation: { temperature: "17-19°C", humidity: "95%", co2: "600 ppm", light: "2,000 lux" },
        fruitingBodyFormation: { temperature: "19-20°C", humidity: "85-92%", co2: "600 ppm", light: "2,000 lux" }
      },
      2: { // Bhutan Oyster
        spawnRun: { temperature: "24-27°C", humidity: "90-100%", co2: "20,000 ppm", light: "Total dark" },
        primordiaFormation: { temperature: "17-19°C", humidity: "95%", co2: "600 ppm", light: "2,000 lux" },
        fruitingBodyFormation: { temperature: "19-20°C", humidity: "85-92%", co2: "600 ppm", light: "2,000 lux" }
      },
      3: { // American Oyster
        spawnRun: { temperature: "24-27°C", humidity: "90-100%", co2: "20,000 ppm", light: "Total dark" },
        primordiaFormation: { temperature: "17-19°C", humidity: "95%", co2: "600 ppm", light: "2,000 lux" },
        fruitingBodyFormation: { temperature: "19-20°C", humidity: "85-92%", co2: "600 ppm", light: "2,000 lux" }
      }
    };
    return params[classId] || params[0];
  };

  return (
    <div className="min-h-screen bg-background dark:bg-slate-950">
      <Navigation />
      
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0">
          <img 
            src={heroImage} 
            alt="Various mushrooms" 
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-primary opacity-80 dark:opacity-90"></div>
        </div>
        
        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center text-primary-foreground">
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              Mushroom Mentor
            </h1>
            <p className="text-xl md:text-2xl mb-8 max-w-3xl mx-auto opacity-90">
              Your intelligent guide to mushroom identification and cultivation. 
              Get tailored recommendations whether you're a farmer or enthusiast.
            </p>
          </div>
        </div>
      </section>

      {/* Mushroom Identification Section */}
      <section id="mushroom-identification" className="py-16 bg-slate-50 dark:bg-slate-900">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">
              Mushroom Identification
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Upload a photo of your mushroom and get detailed growing parameters for optimal cultivation
            </p>
          </div>
          
          <div className="grid gap-8 lg:grid-cols-2">
            {/* Upload Section */}
            <Card className="p-6 bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700">
              <h3 className="text-2xl font-semibold mb-4 flex items-center text-foreground">
                <Upload className="mr-2 h-6 w-6 text-primary" />
                Upload & Analyze
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2 text-foreground">Upload Mushroom Image</label>
                  <div className="border-2 border-dashed border-border rounded-lg p-6 text-center hover:border-primary/50 transition-colors dark:border-slate-600 dark:hover:border-primary/50">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                      ref={fileInputRef}
                    />
                    
                    {uploadedImage ? (
                      <div className="space-y-4">
                        <img 
                          src={uploadedImage} 
                          alt="Uploaded mushroom" 
                          className="max-h-64 mx-auto rounded-lg shadow-soft"
                        />
                        <Button 
                          type="button" 
                          onClick={triggerFileInput}
                          variant="outline"
                        >
                          Change Image
                        </Button>
                      </div>
                    ) : (
                      <div>
                        <Leaf className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                        <Button 
                          type="button" 
                          onClick={triggerFileInput}
                          variant="outline"
                        >
                          Choose Image
                        </Button>
                        <p className="text-sm text-muted-foreground mt-2">
                          PNG, JPG up to 10MB
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                <Button 
                  onClick={analyzeMushroom}
                  disabled={!uploadedImage || isAnalyzing}
                  className="w-full"
                  variant="golden"
                >
                  {isAnalyzing ? "Analyzing..." : "Identify Mushroom"}
                </Button>
              </div>
            </Card>

            {/* Results Section */}
            <Card className="p-6 bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700">
              <h3 className="text-2xl font-semibold mb-4 text-foreground">Results</h3>
              
              {prediction ? (
                <div className="space-y-6">
                  <div className="text-center p-4 bg-accent/10 rounded-lg dark:bg-slate-700/50">
                    <h4 className="text-xl font-bold text-foreground">
                      {prediction.mushroomType?.common || 'Unknown Mushroom'}
                    </h4>
                    <p className="text-sm text-muted-foreground italic mb-2">
                      {prediction.mushroomType?.scientific || 'Scientific name not available'}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Confidence: {prediction.confidence || 'N/A'}% | Class ID: {prediction.classId || 'N/A'}
                    </p>
                    {prediction.features && (
                      <div className="mt-2 text-xs text-muted-foreground">
                        <p>RGB: ({prediction.features.colorMean?.map(v => v.toFixed(1)).join(', ') || 'N/A'})</p>
                        <p>Texture Entropy: {prediction.features.textureFeatures?.entropy?.toFixed(3) || 'N/A'}</p>
                      </div>
                    )}
                  </div>

                  {prediction.growingParameters && (
                    <div>
                      <h5 className="font-semibold text-lg mb-3 flex items-center text-secondary">
                        <Leaf className="mr-2 h-5 w-5" />
                        Growing Parameters
                      </h5>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm border-collapse">
                          <thead>
                            <tr className="bg-accent/10 dark:bg-slate-700/50">
                              <th className="border border-border p-2 text-left dark:border-slate-600">Growth Stage</th>
                              <th className="border border-border p-2 text-left dark:border-slate-600">Temp (°C)</th>
                              <th className="border border-border p-2 text-left dark:border-slate-600">RH (%)</th>
                              <th className="border border-border p-2 text-left dark:border-slate-600">CO2 (ppm)</th>
                              <th className="border border-border p-2 text-left dark:border-slate-600">Light (lux)</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td className="border border-border p-2 font-medium dark:border-slate-600">Spawn Run</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.spawnRun?.temperature || 'N/A'}</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.spawnRun?.humidity || 'N/A'}</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.spawnRun?.co2 || 'N/A'}</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.spawnRun?.light || 'N/A'}</td>
                            </tr>
                            <tr className="bg-accent/5 dark:bg-slate-700/30">
                              <td className="border border-border p-2 font-medium dark:border-slate-600">Primordia Formation</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.primordiaFormation?.temperature || 'N/A'}</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.primordiaFormation?.humidity || 'N/A'}</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.primordiaFormation?.co2 || 'N/A'}</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.primordiaFormation?.light || 'N/A'}</td>
                            </tr>
                            <tr>
                              <td className="border border-border p-2 font-medium dark:border-slate-600">Fruiting Body Formation</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.fruitingBodyFormation?.temperature || 'N/A'}</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.fruitingBodyFormation?.humidity || 'N/A'}</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.fruitingBodyFormation?.co2 || 'N/A'}</td>
                              <td className="border border-border p-2 dark:border-slate-600">{prediction.growingParameters?.fruitingBodyFormation?.light || 'N/A'}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <Leaf className="mx-auto h-16 w-16 mb-4 opacity-50" />
                  <p>Upload an image to get started</p>
                </div>
              )}
            </Card>
          </div>
        </div>
      </section>
      <Footer />
    </div>
  );
};

export default Index;
