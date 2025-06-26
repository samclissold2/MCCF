# Vietnam Power Infrastructure Web Application

This web application transforms your interactive HTML maps into a modern, accessible web dashboard for Vietnam's power infrastructure analysis.

## Features

- **Interactive Dashboard**: Modern web interface with Bootstrap styling
- **Multiple Map Views**: 
  - Power Projects Map
  - Transmission Network Map
  - Substations Map
  - Integrated Infrastructure Map
  - Existing Generators Map
- **Real-time Statistics**: Dynamic loading of project counts and technology distribution
- **API Endpoints**: RESTful API for data access
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Dashboard**:
   Open your browser and go to: `http://localhost:5000`

## Usage

### Dashboard Navigation
- **Home Page**: Overview with statistics and quick access to all maps
- **Projects Map**: View all power generation projects with technology types
- **Transmission Map**: Power transmission infrastructure and grid connections
- **Substations Map**: Electrical substations and their specifications
- **Integrated Map**: Complete infrastructure overview
- **Existing Generators**: Currently operational power facilities

### API Endpoints
- `GET /api/projects`: Returns project data in JSON format
- `GET /projects`: Serves the projects interactive map
- `GET /transmission`: Serves the transmission network map
- `GET /substations`: Serves the substations map
- `GET /integrated`: Serves the integrated infrastructure map
- `GET /existing-generators`: Serves the existing generators map

## Alternative Deployment Options

### Option 2: Streamlit App (Simpler)
If you prefer a simpler setup, you can use Streamlit:

```python
# streamlit_app.py
import streamlit as st
import folium
from map import create_folium_map, read_and_clean_power_data

st.title("Vietnam Power Infrastructure Dashboard")

# Create map
df, name_col = read_and_clean_power_data()
m = create_folium_map(df)

# Display map
st.components.v1.html(m._repr_html_(), height=600)
```

Run with: `streamlit run streamlit_app.py`

### Option 3: Static Site Generator
For a static website that can be hosted on GitHub Pages:

1. **Generate static HTML files**:
   ```python
   # static_generator.py
   from map import *
   import os
   
   # Generate all maps
   maps = {
       'projects': create_folium_map(read_and_clean_power_data()[0]),
       'transmission': create_transmission_map(),
       'substations': create_substation_map(),
       'integrated': create_integrated_map()
   }
   
   # Save to static directory
   for name, m in maps.items():
       m.save(f'static/{name}_map.html')
   ```

2. **Create index.html** with navigation to static files

### Option 4: Docker Container
For easy deployment:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t power-infrastructure-app .
docker run -p 5000:5000 power-infrastructure-app
```

## Customization

### Adding New Maps
1. Create the map function in `map.py`
2. Add a new route in `app.py`
3. Add a new card in `templates/index.html`

### Styling
- Modify CSS in `templates/index.html`
- Add custom Bootstrap themes
- Include additional JavaScript libraries

### Data Sources
- Update data file paths in `config.py`
- Add new data processing functions
- Extend API endpoints for new data types

## Deployment

### Local Development
```bash
python app.py
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Cloud Deployment
- **Heroku**: Add `Procfile` with `web: gunicorn app:app`
- **AWS**: Use Elastic Beanstalk or EC2
- **Google Cloud**: App Engine or Compute Engine
- **Azure**: App Service or Container Instances

## Troubleshooting

### Common Issues
1. **Large file errors**: Ensure all large files are in `.gitignore`
2. **Import errors**: Check all dependencies are installed
3. **Map loading issues**: Verify data file paths in `config.py`
4. **Performance**: Consider caching for large datasets

### Performance Optimization
- Implement caching for map generation
- Use background tasks for heavy computations
- Optimize data loading and processing
- Consider using a CDN for static assets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 