library(shiny)
library(ggplot2)
library(dplyr)

# Load your data
data <- read.csv("D:/assignments/DATA-230/assignment-14/TB_Burden_Country.csv")

# Define the User Interface (UI)
ui <- fluidPage(
  titlePanel("TB Burden Analysis by Country"),
  sidebarLayout(
    sidebarPanel(
      selectInput("year", "Select Year:", choices = unique(data$Year), selected = max(data$Year)),
      selectInput("country", "Select Country:", choices = unique(data$Country.or.territory.name), selected = "Afghanistan")
    ),
    mainPanel(
      plotOutput("barPlot"),
      plotOutput("linePlot"),
      plotOutput("scatterPlot")
    )
  )
)

# Define the Server logic
server <- function(input, output) {
  
  output$barPlot <- renderPlot({
    # Filter data for the selected year and remove rows with NA in TB prevalence
    year_data <- data %>%
      filter(Year == input$year) %>%
      group_by(Country.or.territory.name) %>%
      summarize(TB_Prevalence = mean(Estimated.prevalence.of.TB..all.forms..per.100.000.population, na.rm = TRUE)) %>%
      na.omit()  # Remove rows with NA values
    
    # Select top 20 countries with highest TB prevalence
    top_countries <- year_data %>%
      arrange(desc(TB_Prevalence)) %>%
      slice(1:20)
    
    # Bar plot of TB Prevalence for top 20 countries
    ggplot(top_countries, aes(x = reorder(Country.or.territory.name, TB_Prevalence), y = TB_Prevalence, fill = TB_Prevalence)) + 
      geom_bar(stat = "identity") + 
      coord_flip() +  # Flip for easier reading
      scale_fill_gradient(low = "skyblue", high = "darkblue") +
      ggtitle(paste("Top 20 Countries by TB Prevalence per 100,000 in", input$year)) +
      xlab("Country") + 
      ylab("TB Prevalence per 100,000") +
      theme_minimal()
  })
  
  output$linePlot <- renderPlot({
    # Line plot of TB prevalence over years for the selected country
    country_data <- data %>%
      filter(Country.or.territory.name == input$country) %>%
      group_by(Year) %>%
      summarize(TB_Prevalence = mean(Estimated.prevalence.of.TB..all.forms..per.100.000.population, na.rm = TRUE))
    
    ggplot(country_data, aes(x = Year, y = TB_Prevalence)) + 
      geom_line(color = "dodgerblue", size = 1) + 
      geom_point(color = "darkblue", size = 2) +
      ggtitle(paste("TB Prevalence Over Time in", input$country)) +
      xlab("Year") + 
      ylab("TB Prevalence per 100,000") +
      theme_minimal()
  })
  
  output$scatterPlot <- renderPlot({
    # Scatter plot of TB prevalence vs. case detection rate for each country in the selected year
    scatter_data <- data %>%
      filter(Year == input$year) %>%
      select(Country.or.territory.name, TB_Prevalence = Estimated.prevalence.of.TB..all.forms..per.100.000.population, 
             Case_Detection = Case.detection.rate..all.forms...percent) %>%
      na.omit()  # Remove rows with NA values
    
    ggplot(scatter_data, aes(x = Case_Detection, y = TB_Prevalence, color = TB_Prevalence)) + 
      geom_point(size = 3, alpha = 0.7) + 
      scale_color_gradient(low = "orange", high = "red") +
      ggtitle(paste("TB Prevalence vs. Case Detection Rate in", input$year)) +
      xlab("Case Detection Rate (%)") + 
      ylab("TB Prevalence per 100,000") +
      theme_minimal()
  })
}

# Combine UI and server into an app
shinyApp(ui = ui, server = server)
