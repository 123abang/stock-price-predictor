import React, { useEffect, useState } from "react";
import {
    ChartCanvas,
    Chart,
    XAxis,
    YAxis,
    CandlestickSeries,
    LineSeries,
    discontinuousTimeScaleProvider,
} from "react-financial-charts";

const StockChart = ({ stockData = [] }) => {
    const [width, setWidth] = useState(800);
    const [chartType, setChartType] = useState("candlestick"); // New state for chart type
    const height = 400;

    const processData = (data) => {
        return data
            .map(d => {
                const date = new Date(d.Date || d.date);
                const open = parseFloat(d.Open || d.open);
                const high = parseFloat(d.High || d.high);
                const low = parseFloat(d.Low || d.low);
                const close = parseFloat(d.Close || d.close);

                if (!isNaN(date) && !isNaN(open) && !isNaN(high) && !isNaN(low) && !isNaN(close)) {
                    return { date, open, high, low, close };
                } else {
                    console.warn("Invalid data point:", d);
                    return null;
                }
            })
            .filter(d => d !== null);
    };

    const data = processData(stockData);

    useEffect(() => {
        const handleResize = () => {
            setWidth(window.innerWidth * 0.8);
        };
        handleResize();
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    if (!data || data.length === 0) {
        return <div>Loading data...</div>;
    }

    const xScaleProvider = discontinuousTimeScaleProvider.inputDateAccessor(d => d.date);
    const { data: chartData, xScale, xAccessor, displayXAccessor } = xScaleProvider(data);

    // Toggle chart type between candlestick and line
    const handleToggleChartType = () => {
        setChartType(chartType === "candlestick" ? "line" : "candlestick");
    };

    return (
        <div style={{ width: '100%', height: '400px', margin: 'auto' }}>
            <button onClick={handleToggleChartType} style={{ marginBottom: '10px' }}>
                Use {chartType === "candlestick" ? "Line" : "Candlestick"} Chart
            </button>
            
            <ChartCanvas
                height={height}
                width={width}
                ratio={3}
                margin={{ left: 50, right: 50, top: 10, bottom: 30 }}
                data={chartData}
                xScale={xScale}
                xAccessor={xAccessor}
                displayXAccessor={displayXAccessor}
                seriesName="Stock Chart"
            >
                <Chart id={1} yExtents={d => [d.high, d.low]}>
                    <XAxis />
                    <YAxis />
                    {chartType === "candlestick" ? (
                        <CandlestickSeries />
                    ) : (
                        <LineSeries yAccessor={d => d.close} />
                    )}
                </Chart>
            </ChartCanvas>
        </div>
    );
};

export default StockChart;