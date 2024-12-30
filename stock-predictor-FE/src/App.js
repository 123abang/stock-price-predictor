import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { TrendingUp, Brain, LineChart, Shield, Zap, ArrowRight } from 'lucide-react';
import Stock from './stockTable'
import Chatbot from './chatbot'
import Chat from './chatgpt'


const TypewriterText = ({ text, delay = 50 }) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayText(prev => prev + text[currentIndex]);
        setCurrentIndex(c => c + 1);
      }, delay);

      return () => clearTimeout(timeout);
    }
  }, [currentIndex, text, delay]);

  return <span>{displayText}</span>;
};

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      {/* Hero Section */}
      <header className="bg-indigo-900 text-white">
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-6 w-6" />
              <span className="text-xl font-bold">Options StockAI</span>
            </div>
            <div className="flex space-x-8">
              <a href="/features" className="hover:text-indigo-200">Features</a>
              <a href="#how-it-works" className="hover:text-indigo-200">How It Works</a>
              <a href="/chatbot" className="hover:text-indigo-200">chatbot </a>
            </div>
          </div>
        </nav>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              <TypewriterText text="Predict Stock Markets with AI" />
            </h1>
            <p className="text-xl text-indigo-200 mb-8">
              <TypewriterText 
                text="Advanced machine learning algorithms to help you make smarter investment decisions" 
                delay={30}
              />
            </p>
            <a href="/stocks">
              <button className="bg-indigo-500 hover:bg-indigo-600 text-white px-8 py-3 rounded-lg font-semibold transition-colors">
                  Get Started Free
              </button>
            </a>
            <p className="text-xl text-indigo-200 mb-8 opacity-0 animate-fadeIn" style={{
              animationDelay: '2s',
              animationFillMode: 'forwards'
            }}>
              <TypewriterText 
                text="Advanced machine learning algorithms to help you make smarter investment decisions" 
                delay={30}
              />
            </p>
            <div className="opacity-0 animate-fadeIn" style={{
              animationDelay: '4s',
              animationFillMode: 'forwards'
            }}>
              <a href="/stocks">
                <button className="bg-indigo-500 hover:bg-indigo-600 text-white px-8 py-3 rounded-lg font-semibold transition-colors">
                  Get Started Free
                </button>
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Features Section */}
      <section id="features" className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">Why Choose Options StockAI?</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white p-6 rounded-xl shadow-lg">
              <Brain className="w-12 h-12 text-indigo-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">AI-Powered Analysis</h3>
              <p className="text-gray-600">Advanced machine learning models analyze market patterns and predict trends with high accuracy.</p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-lg">
              <LineChart className="w-12 h-12 text-indigo-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">Real-time Predictions</h3>
              <p className="text-gray-600">Get instant predictions and market insights updated in real-time for informed decision making.</p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-lg">
              <Shield className="w-12 h-12 text-indigo-500 mb-4" />
              <h3 className="text-xl font-semibold mb-2">Risk Management</h3>
              <p className="text-gray-600">Advanced risk assessment tools to protect your investments and optimize returns.</p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <img
                src="https://images.unsplash.com/photo-1642790551116-18e150f248e3?auto=format&fit=crop&q=80&w=800"
                alt="AI Analysis Dashboard"
                className="rounded-lg shadow-lg"
              />
            </div>
            <div className="space-y-6">
              <div className="flex items-start space-x-4">
                <div className="bg-indigo-500 text-white p-2 rounded-full">
                  <Zap className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">Data Collection</h3>
                  <p className="text-gray-600">Our AI continuously collects and analyzes market data, news, and social sentiment.</p>
                </div>
              </div>
              <div className="flex items-start space-x-4">
                <div className="bg-indigo-500 text-white p-2 rounded-full">
                  <Brain className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">AI Processing</h3>
                  <p className="text-gray-600">Advanced algorithms process the data to identify patterns and generate predictions.</p>
                </div>
              </div>
              <div className="flex items-start space-x-4">
                <div className="bg-indigo-500 text-white p-2 rounded-full">
                  <TrendingUp className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">Get Predictions</h3>
                  <p className="text-gray-600">Receive accurate stock predictions and actionable insights directly to your dashboard.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">Simple, Transparent Pricing</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white p-8 rounded-xl shadow-lg">
              <h3 className="text-xl font-semibold mb-4">Basic</h3>
              <p className="text-4xl font-bold mb-6">$0<span className="text-lg text-gray-600">/mo</span></p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-center">
                  <ArrowRight className="w-5 h-5 text-indigo-500 mr-2" />
                  Basic stock predictions
                </li>
                <li className="flex items-center">
                  <ArrowRight className="w-5 h-5 text-indigo-500 mr-2" />
                  Daily updates
                </li>
                <li className="flex items-center">
                  <ArrowRight className="w-5 h-5 text-indigo-500 mr-2" />
                  Email alerts
                </li>
              </ul>
              <button className="w-full bg-indigo-500 hover:bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
                Get Started
              </button>
            </div>
            <div className="bg-indigo-900 text-white p-8 rounded-xl shadow-lg transform scale-105">
              <h3 className="text-xl font-semibold mb-4">Pro</h3>
              <p className="text-4xl font-bold mb-6">$79<span className="text-lg text-indigo-300">/mo</span></p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-center">
                  <ArrowRight className="w-5 h-5 text-indigo-300 mr-2" />
                  Advanced predictions
                </li>
                <li className="flex items-center">
                  <ArrowRight className="w-5 h-5 text-indigo-300 mr-2" />
                  Real-time updates
                </li>
                <li className="flex items-center">
                  <ArrowRight className="w-5 h-5 text-indigo-300 mr-2" />
                  Portfolio management
                </li>
                <li className="flex items-center">
                  <ArrowRight className="w-5 h-5 text-indigo-300 mr-2" />
                  Risk analysis
                </li>
              </ul>
              <button className="w-full bg-white hover:bg-gray-100 text-indigo-900 px-6 py-3 rounded-lg font-semibold transition-colors">
                Get Started
              </button>
            </div>
            <div className="bg-white p-8 rounded-xl shadow-lg">
              <h3 className="text-xl font-semibold mb-4">Enterprise</h3>
              <p className="text-4xl font-bold mb-6">$199<span className="text-lg text-gray-600">/mo</span></p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-center">
                  <ArrowRight className="w-5 h-5 text-indigo-500 mr-2" />
                  Custom stock insights
                </li>
                <li className="flex items-center">
                  <ArrowRight className="w-5 h-5 text-indigo-500 mr-2" />
                  API access
                </li>
                <li className="flex items-center">
                  <ArrowRight className="w-5 h-5 text-indigo-500 mr-2" />
                  Dedicated support
                </li>
              </ul>
              <button className="w-full bg-indigo-500 hover:bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
                Contact Us
              </button>
            </div>
          </div>
        </div>
      </section>
            {/* Financial Insights Section */}
<section id="financial-insights" className="py-20 bg-gray-50">
  <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <h2 className="text-3xl font-bold text-center mb-12">
      Maximize Your Financial Growth
    </h2>
    <div className="grid md:grid-cols-2 gap-12 items-center">
      <div>
        <h3 className="text-xl font-semibold mb-4 text-indigo-900">
          The Power of Growing Stocks
        </h3>
        <p className="text-gray-600 mb-6">
          Investing in the stock market has always been a proven path to building wealth. 
          With growing markets, investors are making significant entries, achieving record-high returns. 
          StockAI empowers you with actionable insights derived from cutting-edge AI algorithms.
        </p>
        <h3 className="text-xl font-semibold mb-4 text-indigo-900">
          Meet StockAI: Your Smart Investment Assistant
        </h3>
        <p className="text-gray-600 mb-6">
          Our AI leverages the latest data, analyzing market trends to predict stock prices 
          with incredible accuracy. Powered by Polygon.io's robust data pipeline, StockAI delivers 
          real-time insights every 2 seconds to keep you ahead in your financial journey.
        </p>
        <p className="text-gray-600">
          <strong className="text-indigo-900">Note:</strong> Stock predictions are based on historical patterns and market behavior 
          but are not guaranteed to be 100% accurate. For the most dynamic experience, 
          subscribe to live data updates.
        </p>
      </div>
      <div>
      <img
        src="https://unsplash.com/photos/Gw_sFen8VhU/download?ixid=MnwzNjUyOXwwfDF8c2VhcmNofDV8fGZpbmFuY2V8ZW58MHx8fHwxNjkzMDg1Mjg2&force=true"
        alt="Person using smartphone and laptop"
        className="rounded-lg shadow-lg max-w-full h-auto"
      />
      </div>
    </div>
  </div>
</section>
      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <TrendingUp className="h-6 w-6" />
                <span className="text-xl font-bold"> Options StockAI</span>
              </div>
              <p className="text-gray-400">Making stock predictions smarter with artificial intelligence.</p>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li>Features</li>
                <li>Pricing</li>
                <li>API</li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-gray-400">
                <li>About</li>
                <li>Blog</li>
                <li>Careers</li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Legal</h4>
              <ul className="space-y-2 text-gray-400">
                <li>Privacy</li>
                <li>Terms</li>
                <li>Security</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-12 pt-8 text-center text-gray-400">
            <p>© 2024 OPtions StockAI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};



const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/stocks" element={<Stock />} />
        <Route path="/chatbot" element={<Chatbot />} />
        <Route path="/chat" element={<Chat />} />
      </Routes>
    </Router>
  );
};

export default App; 
