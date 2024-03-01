USE [Crypto]
GO

/****** Object:  Table [dbo].[universe]    Script Date: 10/9/2021 11:01:06 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[universe](
	[OpenDate] [datetime2](7) NOT NULL,
	[CloseDate] [datetime2](7) NOT NULL,
	[ticker] [nvarchar](50) NOT NULL,
	[Open] [decimal](30, 5) NULL,
	[High] [decimal](30, 5) NULL,
	[Low] [decimal](30, 5) NULL,
	[Close] [decimal](30, 5) NULL,
	[Volume] [decimal](30, 5) NULL,
	[QuoteAssetVolume] [decimal](30, 5) NULL,
	[NumberOfTrades] [decimal](30, 5) NULL,
	[TBBAV] [decimal](30, 5) NULL,
	[TBQAV] [decimal](30, 5) NULL,
	[OpenEpoch] [decimal](15, 2) NOT NULL,
	[CloseEpoch] [decimal](15, 2) NOT NULL
) ON [PRIMARY]
GO


