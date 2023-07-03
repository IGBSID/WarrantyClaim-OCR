SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ImageData](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[api_request_id] [int] NOT NULL,
	[image_url] [nvarchar](max) NULL,
	[image_details] [nvarchar](max) NULL,
	[created_on] [datetime] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
ALTER TABLE [dbo].[ImageData] ADD  CONSTRAINT [PK_ImageData] PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ONLINE = OFF, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
GO
ALTER TABLE [dbo].[ImageData]  WITH CHECK ADD  CONSTRAINT [FK_ImageData_APIRequestData] FOREIGN KEY([api_request_id])
REFERENCES [dbo].[APIData] ([id])
GO
ALTER TABLE [dbo].[ImageData] CHECK CONSTRAINT [FK_ImageData_APIRequestData]
GO
