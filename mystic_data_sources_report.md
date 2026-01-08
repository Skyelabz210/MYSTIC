# Comprehensive Report on Weather and Water Data Sources for MYSTIC Flood Prediction System

## Executive Summary

This report provides a comprehensive analysis of real-time and historical data sources for flood prediction systems, with specific focus on API access requirements, authentication methods, and integration parameters relevant to the MYSTIC flood prediction system. The report covers government, international, and commercial data sources with their technical specifications.

## 1. USGS Water Data Sources

### 1.1 Real-Time Data APIs

**API Endpoints:**
- **IV Data Service**: `https://waterservices.usgs.gov/nwis/iv/` - Real-time streamflow, gage height, precipitation, reservoir levels
- **Daily Values Service**: `https://waterservices.usgs.gov/nwis/dv/` - Daily mean values for surface water and groundwater
- **Groundwater Levels**: `https://waterservices.usgs.gov/nwis/gwlevels/` - Groundwater data retrieval
- **Water Quality Data**: `https://www.waterqualitydata.us/data/` - Water quality portal data

**Authentication Method:**
- No API key required for basic access
- Optional API key available for higher rate limits
- Registration at USGS requires email for advanced access

**Rate Limits:**
- Basic: No explicit limits for individual requests
- With API key: Higher rate limits available
- Recommended: No more than 100 requests per minute

**Data Formats:**
- JSON, WaterML2, CSV, and other formats
- RESTful API responses
- Standard USGS parameter codes for specific measurements

**Cost:**
- Free for all data under USGS Open Data Policy

**Specific Parameters for Flood Prediction:**
- Parameter codes (P_CODES): 00060 for streamflow, 00065 for gage height, 62614 for precipitation
- Site selection by hydrologic unit, state, county
- Real-time data typically available with less than 1-hour latency
- Quality flags and approval status indicators

### 1.2 Specialized USGS Data Systems

**Real-Time Flood Impact (RTFI) Data:**
- Provides flood impact information for infrastructure vulnerability
- Links to nearby USGS real-time streamgages
- Data includes embankments, roads, bridges, pedestrian paths, and buildings

## 2. NOAA Weather Data Sources

### 2.1 National Weather Service API

**API Endpoints:**
- **Base URL**: `https://api.weather.gov/`
- **Grid Points Forecast**: `/gridpoints/{wfo}/{x},{y}/forecast` - Forecast by NWS office/grid coordinates
- **Point Forecast**: `/points/{lat},{lon}` - Forecast for specific coordinates
- **Hourly Forecast**: `/gridpoints/{wfo}/{x},{y}/forecast/hourly`
- **Active Alerts**: `/alerts/active` and `/alerts/active/area/{state}`

**Authentication Method:**
- No API key required
- Required User-Agent header: `User-Agent: app-name (contact-email)`

**Rate Limits:**
- No explicit rate limits published
- Reasonable use policy (typically 5 requests per second)
- Service may return 429 if rate exceeded

**Data Formats:**
- JSON responses following schema.org standards
- GeoJSON for geographic data
- Forecast periods with detailed weather phenomena

**Cost:**
- Free access under NOAA Open Data Policy

**Specific Parameters for Flood Prediction:**
- Precipitation amounts and probability
- Temperature and humidity (affecting evaporation rates)
- Wind speed and direction
- Flood warnings and watches
- Storm surge predictions for coastal areas

### 2.2 National Water Prediction Service (NWPS)

**API Endpoints:**
- **Base URL**: `https://api.water.noaa.gov/nwps/v1/`
- **Streamflow Forecasts**: `/forecast/locations/{location_id}/forecasts`
- **Observations**: `/obs/locations/{location_id}/observations`
- **Location Metadata**: `/metadata/locations`

**Authentication Method:**
- No authentication required for basic access
- Recommended registration for production use

**Rate Limits:**
- No explicit limits published
- Use in compliance with NOAA data policies

**Data Formats:**
- JSON responses with time series data
- Standardized units (cubic feet per second for streamflow)

**Specific Parameters for Flood Prediction:**
- Real-time and forecasted streamflow
- Gauge height observations
- National Water Model outputs
- Probabilistic flood risk indicators

### 2.3 Climate Data Online (CDO)

**API Endpoints:**
- **Base URL**: `https://www.ncei.noaa.gov/cdo-web/api/v2/`
- **Datasets**: `/datasets` - List of available datasets
- **Data Types**: `/datatypes` - Types of data available
- **Location Categories**: `/locationcategories` - Geographic groupings
- **Locations**: `/locations` - Specific location information
- **Data**: `/data` - Actual climate data retrieval

**Authentication Method:**
- Token-based authentication required
- Token obtained from CDO website after registration

**Rate Limits:**
- 5 requests per second per token
- 10,000 requests per day per token

**Data Formats:**
- JSON responses
- Customizable units (standard or metric)

**Specific Parameters for Flood Prediction:**
- Precipitation data (PRECIP_HLY, PRECIP_DLY for hourly/daily)
- Temperature data affecting runoff
- Historical trends for flood probability modeling
- Custom date range selection

## 3. NASA Satellite Data Sources

### 3.1 SMAP (Soil Moisture Active Passive) Mission

**Data Access:**
- **NSIDC DAAC**: `https://nsidc.org/data/smap`
- **Earthdata Login**: Required for access
- **API Access**: Through NASA Earthdata API endpoints

**Authentication Method:**
- Earthdata Login account required
- NASA URS authentication system

**Rate Limits:**
- No specific rate limits for authenticated users
- Standard fair use policies apply

**Data Formats:**
- HDF5, NetCDF, and GeoTIFF formats
- Standardized EASE-Grid 2.0 projections
- L2/L3/L4 soil moisture products

**Specific Parameters for Flood Prediction:**
- Surface soil moisture (0-5 cm depth)
- Root zone soil moisture (0-100 cm depth)
- Freeze/thaw state affecting runoff
- 2-3 day repeat cycle for temporal monitoring
- 9-36 km spatial resolution depending on product

### 3.2 GOES-R Series Satellite Data

**Data Access:**
- **AWS Public Dataset**: `https://registry.opendata.aws/noaa-goes/`
- **Google Cloud Public Dataset**: Accessible via Google Earth Engine
- **NOAA CLASS**: Comprehensive Large Array-data Stewardship System

**Authentication Method:**
- AWS account for S3 bucket access
- Google Cloud account for BigQuery access
- NOAA CLASS registration required

**Rate Limits:**
- AWS S3: Standard AWS limits (typically high)
- Google BigQuery: 1TB querying per month for free tier

**Data Formats:**
- NetCDF for Level 1B and Level 2 products
- GRB2 for numerical weather prediction
- Direct broadcast real-time access via GOES Rebroadcast (GRB)

**Specific Parameters for Flood Prediction:**
- Precipitation estimates from ABI (Advanced Baseline Imager)
- Cloud-top temperatures and heights
- Atmospheric moisture profiles
- Lightning detection from Geostationary Lightning Mapper (GLM)
- Real-time data with 1-5 minute latency

## 4. International Data Sources

### 4.1 ECMWF and Copernicus GloFAS

**API Endpoints:**
- **Open-Meteo GloFAS**: `https://api.open-meteo.com/v1/flood`
- **Copernicus Data Portal**: Registration required for full access
- **ECMWF Web API**: Access through Copernicus Climate Data Store (CDS)

**Authentication Method:**
- No authentication for Open-Meteo non-commercial use
- API key required for commercial use
- ECMWF/CDS requires account registration

**Rate Limits:**
- Open-Meteo: Not explicitly limited for non-commercial use
- Commercial use: As per agreement terms

**Data Formats:**
- JSON responses with time series
- Standardized discharge measurements (m³/s)
- Ensemble forecast with 52 members available

**Specific Parameters for Flood Prediction:**
- River discharge forecasts up to 15 days ahead
- Historical river discharge from 1984-present
- 5 km spatial resolution (0.05°)
- Statistical analysis (mean, median, percentiles)
- Flood probability estimates

### 4.2 European Flood Awareness System (EFAS)

**Data Access:**
- **EFAS Portal**: Registration required
- **EUMETNET**: European meteorological network data
- **Real-time ensemble forecasting system**

**Authentication Method:**
- Registration with European meteorological services
- Institutional access typically required

**Specific Parameters for Flood Prediction:**
- Multi-model ensemble forecasts
- 6-15 day medium-range forecasts
- European river basins coverage
- Probabilistic flood guidance

## 5. Oceanographic and Space Weather Data

### 5.1 NOAA CO-OPS (Center for Operational Oceanographic Products and Services)

**API Endpoints:**
- **Base URL**: `https://api.tidesandcurrents.noaa.gov/api/prod/`
- **Water Level Data**: `/water_level` - Verified water level observations
- **High/Low Tide Predictions**: `/high_low` - Predicted high/low waters
- **Currents**: `/currents` - Ocean current measurements
- **Meteorology**: `/meteorology` - Wind, barometric pressure, air/water temperature

**Authentication Method:**
- No API key required
- Required: Application name and email in request parameters

**Rate Limits:**
- 100 requests per hour for basic access
- Higher limits available for registered applications

**Specific Parameters for Flood Prediction:**
- Barometric pressure (affects storm surge)
- Wind speed and direction (affects coastal flooding)
- Water levels and tidal predictions
- Storm surge guidance

### 5.2 NOAA National Data Buoy Center (NDBC)

**Data Access:**
- **Real-time Data**: Available via multiple formats
- **Historical Data**: Archive access through FTP and web interfaces
- **Station Information**: Metadata about buoy locations and sensors

**Authentication Method:**
- No authentication required
- Open access policy

**Specific Parameters for Flood Prediction:**
- Barometric pressure readings
- Wind speed and direction
- Wave heights and periods
- Sea surface temperatures

### 5.3 Space Weather Data

**API Endpoints:**
- **SWPC (Space Weather Prediction Center)**: `https://services.swpc.noaa.gov/`
- **Kp Index**: `https://services.swpc.noaa.gov/json/Kp.json`
- **Solar Wind Data**: `https://services.swpc.noaa.gov/json/solar_wind/`

**Authentication Method:**
- No authentication required
- Open data policy

**Specific Parameters for Flood Prediction:**
- Kp index (geomagnetic activity indicator)
- Solar wind parameters (speed, density, temperature)
- Magnetic field components
- Potential impacts on atmospheric conditions

## 6. NEXRAD Radar Data

**Data Access:**
- **AWS Public Dataset**: `https://registry.opendata.aws/noaa-nexrad/`
- **NOAA NCEI**: Archive access through Level II and Level III data
- **Real-time Access**: Via NOAAPORT or direct broadcast

**Authentication Method:**
- AWS account for S3 access
- No authentication for public AWS datasets

**Data Formats:**
- Level II: Raw radar data in WMO formats
- Level III: Product data including precipitation estimates
- NetCDF and HDF5 formats available

**Specific Parameters for Flood Prediction:**
- Precipitation rate estimates
- Precipitation accumulation products
- Storm tracking information
- Reflectivity and velocity data for storm analysis
- 4-6 minute data availability in real-time

## 7. Commercial Weather Data APIs

### 7.1 OpenWeatherMap

**API Endpoints:**
- **Current Weather**: `https://api.openweathermap.org/data/2.5/weather`
- **Historical**: `https://api.openweathermap.org/data/2.5/history/city`
- **Precipitation**: Through various endpoints

**Authentication Method:**
- API key required
- Free tier with limited requests
- Paid plans for higher rate limits

**Rate Limits:**
- Free: 1,000 calls per day
- Paid plans: Up to 60 calls/minute

**Specific Parameters for Flood Prediction:**
- Current precipitation data
- Rainfall history and forecasts
- Temperature and humidity
- Pressure data

### 7.2 WeatherAPI

**API Endpoints:**
- **Current Weather**: `/current.json`
- **Forecast**: `/forecast.json`
- **Historical**: `/history.json`
- **Real-time**: `/realtime.json`

**Authentication Method:**
- API key required
- Free and paid plans available

**Specific Parameters for Flood Prediction:**
- Real-time precipitation
- Weather alerts and warnings
- Historical weather patterns
- UV and air quality data

## 8. Data Integration Recommendations for MYSTIC System

### 8.1 Priority Data Sources
1. **USGS Real-time Streamflow**: Highest priority for flood detection
2. **NOAA NWPS**: Critical for forecast integration
3. **NOAA Weather API**: For weather warnings and conditions
4. **NEXRAD Data**: Essential for precipitation monitoring
5. **NASA GOES**: For real-time satellite precipitation estimates

### 8.2 Implementation Strategy
1. **Real-time Pipeline**: Integrate USGS, NWPS, and NEXRAD for immediate monitoring
2. **Forecast Integration**: Use NOAA forecast services and ECMWF GloFAS
3. **Historical Baseline**: Leverage CDO and USGS historical data for model training
4. **Satellite Enhancement**: Add NASA SMAP soil moisture and GOES precipitation data

### 8.3 Technical Considerations
- Implement asynchronous data fetching to handle different update frequencies
- Include quality control and validation for each data source
- Design fallback mechanisms when primary sources are unavailable
- Consider data caching to reduce API load and improve response times
- Implement proper error handling for network issues and API failures

## Conclusion

The MYSTIC flood prediction system has access to numerous high-quality data sources from government, international, and commercial providers. The key to effective flood prediction lies in the intelligent combination of real-time streamflow data from USGS, precipitation forecasts from NOAA, satellite data from NASA, and international flood forecasting systems like ECMWF GloFAS. The system should prioritize real-time data integration while maintaining historical data access for model validation and improvement.