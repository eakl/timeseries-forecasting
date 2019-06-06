setwd("~/dev/case")

########################################################################################
### Install & Load Packages
########################################################################################

#install.packages('devtools')
require(devtools)

#install_github('rstudio/keras')
require(keras)

if(!require('pacman')) install.packages('pacman')
pacman::p_load(readr, tibble, magrittr, lubridate ,dplyr ,tidyr, mltools, tseries)

#install_keras()

########################################################################################
## Functions
########################################################################################

##### Load
load <- function(country) {
  ## Load
  raw <- read_csv(
    './forecast_data.csv'
    ,col_names = T
    ,skip = 1
    ,col_types = cols(
      `Made On Date` = col_date(format = '%d-%m-%Y')
      ,`Gross Reservations` = col_double()
      ,`Gross Reservations with Vouchers` = col_double()
      ,`Merchants` = col_double()
      ,`Reservations with 30%+ Discounts` = col_double()
      ,`Same day Dining Bookings` = col_double()
      ,`Gross Reservations_1` = col_double()
      ,`Gross Reservations with Vouchers_1` = col_double()
      ,`Merchants_1` = col_double()
      ,`Reservations with 30%+ Discounts_1` = col_double()
      ,`Same day Dining Bookings_1` = col_double()
    ))

  ## Clean
  raw_c <- raw %>%
    rename(
      date = `Made On Date`
      ,gross_reservations_sg = `Gross Reservations`
      ,w_vouchers_sg = `Gross Reservations with Vouchers`
      ,merchants_sg = `Merchants`
      ,w_30p_pp_off_sg = `Reservations with 30%+ Discounts`
      ,same_day_sg = `Same day Dining Bookings`
      ,gross_reservations_th = `Gross Reservations_1`
      ,w_vouchers_th = `Gross Reservations with Vouchers_1`
      ,merchants_th = `Merchants_1`
      ,w_30p_pp_off_th = `Reservations with 30%+ Discounts_1`
      ,same_day_th = `Same day Dining Bookings_1`
    )

  if(country == 'sg') {
    ## Extract
    data <- raw_c %>%
      select(date, gross_reservations_sg, w_vouchers_sg) %>%
      rename(resa = gross_reservations_sg, vouchers = w_vouchers_sg)
  } else if (country == 'th') {
    data <- raw_c %>%
      select(date, gross_reservations_th, w_vouchers_th) %>%
      rename(resa = gross_reservations_th, vouchers = w_vouchers_th)
  } else {
    stop('provide a country')
  }

  return(data)
}

##### Transform serie
transform <- function(serie) {
  l <- log(serie) # Log
  d <- diff(l) # Diff 1 time
  s <- scale(d) # Normalize

  return(
    list(
      log = l
      ,diff = d
      ,norm = s[,1]
      ,center = attr(s, 'scaled:center')
      ,scale = attr(s, 'scaled:scale')
      ,init = l[1]
    )
  )
}

##### Untransform serie
untransform <- function(serie, scale=1, center=0, init=0) {
  d <- ((serie * scale) + center) # Un-scale
  l <- diffinv(d, xi = init) # Un-diff
  o <- exp(l) # Un-log

  return(o)
}

# Split dataset
split_dataset <- function(data, n_test) {
  n_hist <- (dim(data)[1] - n_test)

  return(
    list(
      hist = data[1:n_hist,]
      ,test = data[(n_hist+1):(n_hist+n_test),]
    )
  )
}

# From time series to supervised learning data shape
to_supervised <- function(data, n_in, n_out) {
  x <- list()
  y <- list()

  samples <- dim(data)[1]
  features <- dim(data)[2]

  # We create a rolling window for inputs and outputs
  for (i in 1:samples) {
    in_end = i + (n_in - 1)
    out_end = in_end + n_out

    if (out_end <= samples) {
      x[[i]] <- hist[i:in_end,]
      y[[i]] <- hist[(in_end+1):out_end,1]
    }
  }

  x_array <- array(unlist(x), dim = c(n_in, features, length(x)))
  y_array <- array(unlist(y), dim = c(n_out, 1, length(y)))

  return (
    list(
      x = apply(x_array, c(1,2), t)
      ,y = apply(y_array, c(1,2), t)
    )
  )
}

# Shape data for the LSTM model
prepare_data <- function(data, split_train) {
  n_train <- round(dim(s_hist$x)[1]*(split_train))
  n_val <- round(dim(s_hist$x)[1]*(1-split_train))

  return (
    list(
      tx = array(s_hist$x[1:n_train,,], dim = c(n_train, dim(s_hist$x)[2], dim(s_hist$x)[3]))
      ,ty = array(s_hist$y[1:n_train,,], dim = c(n_train, dim(s_hist$y)[2], dim(s_hist$y)[3]))
      ,vx = array(s_hist$x[(n_train+1):(n_train+n_val),,], dim = c(n_val, dim(s_hist$x)[2], dim(s_hist$x)[3]))
      ,vy = array(s_hist$y[(n_train+1):(n_train+n_val),,], dim = c(n_val, dim(s_hist$y)[2], dim(s_hist$y)[3]))
    )
  )
}

##### Build model
build_model <- function(data, save=FALSE) {

  train_x <- data$tx
  train_y <- array(data$ty, dim = c(dim(data$ty)[1], dim(data$ty)[2]))
  val_x <- data$vx
  val_y <- array(data$vy, dim = c(dim(data$vy)[1], dim(data$vy)[2]))

  n_timesteps <- dim(train_x)[2]
  n_features <- dim(train_x)[3]
  n_outputs <- dim(train_y)[2]

  if(file.exists('./model.h5')) {
    model <- load_model_hdf5('model.h5')
  } else {
    epochs <- 250
    batch_size <- 20

    model <- keras_model_sequential() %>%
      layer_dense(
        units = 60
        ,activation = 'softmax'
        ,input_shape = c(n_timesteps, n_features)
      ) %>%
      layer_dense(
        units = 10
        ,activation = 'softmax'
      ) %>%
      layer_flatten() %>%
      layer_dense(
        units = 1
        ,activation = 'linear'
      )

    model %>% compile(
      optimizer = optimizer_rmsprop(lr = 0.001),
      loss = 'mse',
      metrics = 'accuracy'
    )

    result <- model %>% fit(
      train_x,
      train_y,
      epochs = epochs,
      batch_size = batch_size,
      validation_data = list(val_x, val_y),
      verbose = 2,
      shuffle = FALSE
    )

    if(save == TRUE) {
      model %>% save_model_hdf5('model.h5')
    }
  }

  return(
    list(
      model = model,
      result = result
    )
  )
}

##### Make predictions
make_prediction <- function(model, hist, test, n_in, n_out) {
  pred_y <- list()

  input_ <- hist[(dim(hist)[1]-n_in+1):(dim(hist)[1]),]
  input <- array(input_, dim = c(1, dim(input_)[1], dim(input_)[2]))

  # Reservations are predicted recusrsively along with vouchers utilization
  for (i in 1:(n_out-1)) {
    yhat <- model %>% predict(input) %>% .[1,1]
    pred_y[[i]] <- yhat
    input <- array(
      c(c(input[,,1][-1], yhat), c(input[,,2][-1], test[i, 2]))
      ,dim = c(1, n_in, dim(test)[2])
    )
  }

  pred_y <- unlist(pred_y)

  return(pred_y)
}

##### Get original data back
get_original <- function(hist, pred, trans) {
  vec <- c(hist[,1], pred)

  o <- untransform(vec, trans$scale, trans$center, trans$init)

  return(o[(length(o)-length(pred)):length(o)])
}

##### Accuracy test
test_accuracy <- function(actual, pred) {
  error_ <- list()
  rmse_ <- list()
  mae_ <- list()
  ape_ <- list()

  for(i in 1:length(actual)) {
    error_[[i]] <- actual[i]-pred[i]
    rmse_[[i]] <- sqrt(mse(pred[i], actual[i]))
    mae_[[i]] <- abs(actual[i]-pred[i])
    ape_[[i]] <- abs(actual[i]-pred[i])/actual[i]
  }

  return(
    list(
      all = list(
        error = sum(actual-pred)
        ,rmse = rmse(pred, actual)
        ,mae = mean(abs(actual-pred))
        ,ape = sum(abs(actual-pred))/sum(actual)
      )
      ,each = list(
        error = unlist(error_)
        ,rmse = unlist(rmse_)
        ,mae = unlist(mae_)
        ,ape = unlist(ape_)
      )
    )
  )
}

########################################################################################
## Run
########################################################################################

set.seed(123)

##### Variables

n_inputs <- 30
n_outputs <- 1
n_test <- 30
train_split = .8

##### Load and transform raw data

raw <- load('th')

# We set vouchers utilization at minimum 1 (because of the log function we will apply later)
c_raw <- raw %>%
  select(-date) %>%
  mutate(vouchers = ifelse(vouchers == 0, 1, vouchers)) %>%
  as.matrix %>%
  unname

plot(c_raw[,1], type='l')

# Split dataset into train and test sets
# The test set is the 30 days we aim at predicting

dataset <- split_dataset(c_raw, n_test)

# The time series are non stationary. We need to transform them
# We log, difference and scale each time series
# Trend and seasonality can be handled by the LSTM. Don't need to de-trend and de-seasonalize

plot(stl(ts(c_raw[,1], f=7), s.window = 'periodic'))

t_hist_resa <- transform(dataset$hist[,1])
t_hist_voucher <- transform(dataset$hist[,2])
t_test_resa <- transform(dataset$test[,1])
t_test_voucher <- transform(dataset$test[,2])

# We get a stationary time series with (almost) constant mean and variance

hist <- array(c(t_hist_resa$norm, t_hist_voucher$norm), dim = c(dim(dataset$hist)[1], dim(dataset$hist)[2]))
test <- array(c(t_test_resa$norm, t_test_voucher$norm), dim = c(dim(dataset$test)[1], dim(dataset$test)[2]))

plot(stl(ts(hist[,1], f=7), s.window = 'periodic'))
#plot(mstl(ts(hist[,1], f=7)))

##### Time serie to supervised learning

# We transform the data into a 3d tensor of shape (samples, timestep, features)
# Here, (791, 30, 2) for the input and (791, 1, 1) for the output
# We use the 30 last days of the time series to predict the next day

s_hist <- to_supervised(hist, n_inputs, n_outputs)

##### Building the model

# Put data in the right format for the LSTM model
# The train set will represent 80% of the data. The remaining 20% will be use as the validation set

m_data <- prepare_data(s_hist, train_split)

# We build and train the model on a MLP
# Inputs: 30 previous days
# Outputs: next day

model <- build_model(m_data, save=FALSE)

##### Predicting the next day 30 times recursively

predictions <- make_prediction(model$model, hist, test, n_inputs, 30)

##### Converting back the predictions to original scale

pred <- get_original(hist, predictions, t_hist_resa)

# Prediction plot
a <- dataset$hist[,1]
t <- c(rep(NA, length(dataset$hist[,1])), dataset$test[,1])
p <- c(rep(NA, length(dataset$hist[,1])), round(pred))

plot(a, type = 'l')
lines(t, type = 'l', col = 'blue')
lines(p, type = 'l', col = 'green')

# Accuracy
metrics <- test_accuracy(dataset$test[,1], round(pred))
