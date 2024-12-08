import React, { useEffect, useState } from 'react';
import axios from 'axios';
import StockChart from './stockChart';
import { motion } from 'framer-motion';
import { ArrowUpCircle, ArrowDownCircle, Search } from 'lucide-react';

const TypewriterText = ({ text, className, delay = 0 }) => {
    const [displayedText, setDisplayedText] = useState('');
    const [currentIndex, setCurrentIndex] = useState(0);
  
    useEffect(() => {
        if (currentIndex < text.length) {
            const timeout = setTimeout(() => {
                setDisplayedText(prev => prev + text[currentIndex]);
                setCurrentIndex(prev => prev + 1);
            }, 100);
    
            return () => clearTimeout(timeout);
        }
    }, [currentIndex, text]);
  
    return (
        <motion.span
            className={className}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay }}
        >
            {displayedText}
            <span className="animate-pulse">|</span>
        </motion.span>
    );
};

const StockTable = () => {
    const [stocks, setStocks] = useState([]);
    const [search, setSearch] = useState('');
    const [selectedStock, setSelectedStock] = useState(null);
    const [stockDetails, setStockDetails] = useState(null);

    useEffect(() => {
        const fetchStocks = async () => {
            try {
                const response = await axios.get('http://127.0.0.1:5000/api/stocks');
                setStocks(response.data);
            } catch (error) {
                console.error('Error fetching stock data:', error);
            }
        };
        fetchStocks();
    }, []);

    const filteredStocks = stocks.filter(stock =>
        stock.T.toLowerCase().includes(search.toLowerCase())
    );

    const handleRowClick = async (ticker) => {
        try {
            const detailResponse = await axios.get(`http://127.0.0.1:5000/api/stock/${ticker}/details`);
            setStockDetails(detailResponse.data);
            const chartResponse = await axios.get(`http://127.0.0.1:5000/api/stock/${ticker}`);
            setSelectedStock(chartResponse.data);
        } catch (error) {
            console.error('Error fetching stock details:', error);
        }
    };

    const renderArrow = (currentPrice, previousClose) => {
        if (!currentPrice || !previousClose) return null;
        return currentPrice > previousClose ? (
            <ArrowUpCircle className="text-green-500 w-6 h-6" />
        ) : (
            <ArrowDownCircle className="text-red-500 w-6 h-6" />
        );
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-8">
            <motion.div 
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="max-w-7xl mx-auto"
            >
                <h1 className="text-4xl font-bold text-center text-gray-800 mb-8">
                    <TypewriterText text="Stock Market Dashboard" />
                </h1>
                
                <div className="text-center text-gray-600 mb-8">
                    <TypewriterText 
                        text="Stock data from January 1, 2023, to December 31, 2023"
                        delay={2.5} // Starts after the first text finishes
                    />
                </div>

                <div className="relative mb-8">
                    <Search className="absolute left-4 top-3 text-gray-400" />
                    <input
                        type="text"
                        placeholder="Search stocks..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="w-full pl-12 pr-4 py-3 rounded-xl border border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all duration-300"
                    />
                </div>

                <div className="bg-white rounded-xl shadow-lg overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead className="bg-gray-50">
                                <tr>
                                    {['Ticker', 'Price', 'Volume', 'Open', 'Close', 'High', 'Low', 'Change'].map((header) => (
                                        <th key={header} className="px-6 py-4 text-left text-sm font-semibold text-gray-600">
                                            {header}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {filteredStocks.map((stock, index) => (
                                    <motion.tr
                                        key={index}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: index * 0.1 }}
                                        onClick={() => handleRowClick(stock.T)}
                                        className="border-t border-gray-100 hover:bg-blue-50 cursor-pointer transition-colors duration-150"
                                    >
                                        <td className="px-6 py-4 font-semibold text-blue-600">{stock.T}</td>
                                        <td className="px-6 py-4">${stock.c.toFixed(2)}</td>
                                        <td className="px-6 py-4">{stock.v.toLocaleString()}</td>
                                        <td className="px-6 py-4">${stock.o.toFixed(2)}</td>
                                        <td className="px-6 py-4">${stock.c.toFixed(2)}</td>
                                        <td className="px-6 py-4">${stock.h.toFixed(2)}</td>
                                        <td className="px-6 py-4">${stock.l.toFixed(2)}</td>
                                        <td className="px-6 py-4">{renderArrow(stock.c, stock.o)}</td>
                                    </motion.tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {selectedStock && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-8 bg-white p-6 rounded-xl shadow-lg"
                    >
                        <h3 className="text-2xl font-bold text-gray-800 mb-4">
                            {selectedStock.ticker} Stock Chart
                        </h3>
                        <StockChart stockData={selectedStock.data} />
                    </motion.div>
                )}

                {stockDetails && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-8 bg-white p-6 rounded-xl shadow-lg"
                    >
                        <h3 className="text-2xl font-bold text-gray-800 mb-4">
                            {stockDetails.name} ({stockDetails.ticker})
                        </h3>
                        <div className="grid gap-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div className="p-4 bg-gray-50 rounded-lg">
                                    <p className="text-sm text-gray-500">Industry</p>
                                    <p className="font-semibold">{stockDetails.industry}</p>
                                </div>
                                <div className="p-4 bg-gray-50 rounded-lg">
                                    <p className="text-sm text-gray-500">Sector</p>
                                    <p className="font-semibold">{stockDetails.sector}</p>
                                </div>
                                <div className="p-4 bg-gray-50 rounded-lg">
                                    <p className="text-sm text-gray-500">Market Cap</p>
                                    <p className="font-semibold">{stockDetails.market_cap}</p>
                                </div>
                                <div className="p-4 bg-gray-50 rounded-lg">
                                    <p className="text-sm text-gray-500">Employees</p>
                                    <p className="font-semibold">{stockDetails.employees}</p>
                                </div>
                            </div>
                            <div className="p-4 bg-gray-50 rounded-lg">
                                <p className="text-sm text-gray-500 mb-2">Description</p>
                                <p className="text-gray-700">{stockDetails.description}</p>
                            </div>
                        </div>
                    </motion.div>
                )}
            </motion.div>
        </div>
    );
};

export default StockTable;